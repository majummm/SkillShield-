"""
Nucleo de IA do SkillShield.

Este modulo concentra tudo que NAO depende de requisicao HTTP:
- banco de cenarios (50 cenarios sinteticos, 5 categorias)
- geracao do dataset sintetico de treinamento (por regras coerentes)
- treinamento e selecao do modelo preditivo (Regressao Logistica / Random
  Forest / Gradient Boosting, com XGBoost se disponivel)
- calculo de Perfil de Vulnerabilidade e SSI
- motor de treinamento adaptativo (escolha de categoria/dificuldade)

E' o mesmo nucleo cientifico validado no notebook SkillShield_Nucleo_IA.ipynb,
reorganizado para ser importado por um servidor web (app.py).

IMPORTANTE (transparencia cientifica): o modelo preditivo e treinado com
dados SINTETICOS gerados por regras (nao ha usuarios reais nesses dados de
treino). As metricas de avaliacao do modelo (accuracy, F1 etc.) refletem
apenas esse dataset sintetico e sao recalculadas a cada vez que o servidor
inicia. Substitua `generate_dataset()` por dados reais coletados assim que
houver volume suficiente.
"""

import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Caracteristicas, categorias e pesos de relevancia (Secao 5/6 do notebook)
# ---------------------------------------------------------------------------
FEATURES = [
    "identified_risk",
    "protected_data",
    "avoided_sharing",
    "used_anonymization",
    "checked_policy",
    "verified_information",
    "recognized_social_engineering",
    "questioned_ai",
    "considered_intellectual_property",
]

CATEGORIES = [
    "Dados pessoais",
    "Informacoes confidenciais",
    "Engenharia social",
    "Confiabilidade e limitacoes da IA",
    "Propriedade intelectual e direitos autorais",
]

FEATURE_QUESTIONS = {
    "identified_risk": "Percebeu que a situacao envolve algum risco relevante?",
    "protected_data": "Evitaria expor dados pessoais sensiveis?",
    "avoided_sharing": "Evitaria compartilhar a informacao sem cuidados adicionais?",
    "used_anonymization": "Consideraria anonimizar/mascarar informacoes antes de usar a IA?",
    "checked_policy": "Verificaria a politica da empresa antes de agir?",
    "verified_information": "Verificaria a confiabilidade da informacao/fonte antes de usa-la?",
    "recognized_social_engineering": "Reconhece sinais de possivel golpe/engenharia social?",
    "questioned_ai": "Questionaria/verificaria a resposta da IA antes de confiar nela?",
    "considered_intellectual_property": "Consideraria direitos autorais/propriedade intelectual?",
}

CATEGORY_FEATURE_WEIGHTS = {
    "Dados pessoais": {
        "identified_risk": 1.2, "protected_data": 1.5, "avoided_sharing": 1.3,
        "used_anonymization": 1.4, "checked_policy": 1.0, "verified_information": 0.6,
        "recognized_social_engineering": 0.3, "questioned_ai": 0.5, "considered_intellectual_property": 0.3,
    },
    "Informacoes confidenciais": {
        "identified_risk": 1.3, "protected_data": 1.1, "avoided_sharing": 1.5,
        "used_anonymization": 1.0, "checked_policy": 1.4, "verified_information": 0.6,
        "recognized_social_engineering": 0.4, "questioned_ai": 0.5, "considered_intellectual_property": 0.4,
    },
    "Engenharia social": {
        "identified_risk": 1.2, "protected_data": 0.5, "avoided_sharing": 0.6,
        "used_anonymization": 0.2, "checked_policy": 0.7, "verified_information": 0.9,
        "recognized_social_engineering": 1.6, "questioned_ai": 0.8, "considered_intellectual_property": 0.2,
    },
    "Confiabilidade e limitacoes da IA": {
        "identified_risk": 0.9, "protected_data": 0.3, "avoided_sharing": 0.3,
        "used_anonymization": 0.2, "checked_policy": 0.5, "verified_information": 1.5,
        "recognized_social_engineering": 0.5, "questioned_ai": 1.6, "considered_intellectual_property": 0.3,
    },
    "Propriedade intelectual e direitos autorais": {
        "identified_risk": 0.9, "protected_data": 0.3, "avoided_sharing": 0.5,
        "used_anonymization": 0.2, "checked_policy": 1.0, "verified_information": 0.7,
        "recognized_social_engineering": 0.2, "questioned_ai": 0.6, "considered_intellectual_property": 1.6,
    },
}

RISK_ORDER = ["Baixo", "Medio", "Alto"]
DIFFICULTY_ORDER = ["Basica", "Intermediaria", "Avancada"]

# ---------------------------------------------------------------------------
# Banco de 50 cenarios (10 por categoria)
# ---------------------------------------------------------------------------


def build_scenarios():
    scenarios = []
    sid = 1

    dados_pessoais = [
        ("Basica", "Um colega pede ajuda para organizar uma lista de contatos de clientes (nome, telefone, e-mail) usando uma ferramenta de IA online.", "Voce usaria a ferramenta de IA diretamente com a lista completa?", "Vazamento de dados de contato de clientes para terceiros via ferramenta externa.", ["reconhecer que ha dados pessoais", "evitar inserir dados reais sem necessidade", "considerar anonimizacao ou dados fake para teste", "verificar politica de uso de ferramentas externas"]),
        ("Intermediaria", "Seu gestor enviou uma planilha contendo nomes, CPFs e salarios dos funcionarios e pediu que voce utilize uma ferramenta de IA para gerar um relatorio.", "O que voce faria?", "Exposicao de dados sensiveis de RH (CPF, salario) em plataforma externa nao autorizada.", ["reconhecer que existem dados pessoais", "evitar compartilhamento direto", "considerar anonimizacao", "considerar politicas da organizacao", "avaliar se a ferramenta utilizada e autorizada"]),
        ("Avancada", "Voce precisa treinar um assistente de IA interno com historico de atendimentos que contem dados de saude de clientes.", "Como voce prepararia esses dados antes de usa-los?", "Uso de dados sensiveis de saude sem consentimento ou anonimizacao adequada, violando LGPD.", ["identificar dado sensivel (saude)", "aplicar anonimizacao/pseudonimizacao", "verificar base legal de tratamento", "checar politica de retencao de dados"]),
        ("Basica", "Voce quer usar uma IA para escrever uma carta de recomendacao e pretende colar o curriculo completo do candidato, incluindo endereco residencial.", "Voce inclui o endereco residencial no prompt?", "Compartilhamento desnecessario de dado pessoal (endereco) com terceiro (provedor de IA).", ["identificar dados desnecessarios ao objetivo", "remover informacoes nao essenciais", "minimizar dados enviados"]),
        ("Intermediaria", "Um cliente pede que voce use uma IA para transcrever uma ligacao de atendimento que contem numero de cartao de credito mencionado verbalmente.", "Como voce conduziria a transcricao?", "Exposicao de dado financeiro sensivel a um servico de transcricao por IA de terceiros.", ["identificar dado financeiro sensivel", "redigir/mascarar a informacao antes de processar", "verificar contrato de confidencialidade do fornecedor de IA"]),
        ("Avancada", "Sua empresa quer construir um chatbot de RH usando IA generativa alimentado com o banco de dados completo de colaboradores.", "Que cuidados voce recomendaria antes de liberar o projeto?", "Vazamento massivo de dados pessoais de todos os colaboradores atraves de respostas do chatbot.", ["propor anonimizacao/tokenizacao", "restringir escopo de dados acessiveis pelo modelo", "implementar controle de acesso e auditoria", "validar conformidade com a LGPD"]),
        ("Basica", "Voce recebeu uma foto de um documento de identidade de um colega e quer usar IA para 'melhorar a qualidade da imagem'.", "Voce envia a foto do documento para uma ferramenta de IA online desconhecida?", "Exposicao de documento de identificacao pessoal a terceiro nao confiavel.", ["reconhecer sensibilidade do documento", "evitar uso de ferramentas nao verificadas", "buscar alternativa segura ou autorizada pela empresa"]),
        ("Intermediaria", "Voce esta usando um assistente de IA para responder e-mails e ele sugere preencher automaticamente dados de clientes a partir de um banco integrado.", "Voce aceita a sugestao sem revisar quais dados serao enviados?", "Envio automatico de dados pessoais sem revisao humana, ampliando a superficie de exposicao.", ["revisar dados antes do envio automatico", "questionar quais dados o assistente tem acesso", "configurar permissoes minimas necessarias"]),
        ("Avancada", "Um pesquisador quer usar uma base de dados publica de pacientes (supostamente anonimizada) para treinar um modelo de IA proprio.", "Quais verificacoes voce faria antes de utilizar essa base?", "Reidentificacao de individuos a partir de dados mal anonimizados (risco de re-identificacao).", ["avaliar qualidade da anonimizacao", "checar risco de reidentificacao cruzando bases", "verificar autorizacao legal/etica de uso"]),
        ("Basica", "Voce quer pedir para uma IA generativa criar um cartao de aniversario personalizado para um colega, usando o nome completo e a data de nascimento dele.", "Isso representa algum risco de dado pessoal?", "Compartilhamento de dado pessoal (data de nascimento) sem necessidade real com servico externo.", ["avaliar se o dado e realmente necessario", "considerar usar apenas o primeiro nome", "checar politica de privacidade da ferramenta"]),
    ]

    confidenciais = [
        ("Basica", "Voce quer resumir uma ata de reuniao interna que menciona uma futura fusao entre empresas ainda nao divulgada publicamente.", "Voce cola a ata inteira em uma ferramenta de IA publica para gerar o resumo?", "Vazamento de informacao estrategica confidencial (fusao) para fora da empresa.", ["reconhecer o carater confidencial da informacao", "evitar ferramentas publicas nao homologadas", "verificar politica de confidencialidade/NDA"]),
        ("Intermediaria", "Um fornecedor pede que voce compartilhe, via chat com IA, detalhes do contrato comercial vigente para 'agilizar' uma proposta.", "Como voce responderia a esse pedido?", "Divulgacao de termos contratuais confidenciais a terceiros nao autorizados.", ["identificar informacao confidencial contratual", "recusar compartilhamento nao autorizado", "encaminhar para canal formal/juridico"]),
        ("Avancada", "Voce esta usando um assistente de IA de codigo para revisar um repositorio interno que contem chaves de API e segredos de producao.", "Que cuidados voce toma antes de enviar o codigo para analise?", "Exposicao de credenciais/segredos de producao a um servico de IA externo.", ["remover ou mascarar segredos antes do envio", "usar ferramentas homologadas com garantias contratuais", "rotacionar credenciais caso ja tenham sido expostas"]),
        ("Basica", "Voce recebeu um documento marcado como 'Confidencial - uso interno' e quer pedir a uma IA que o traduza para ingles.", "Voce usa uma ferramenta de traducao online gratuita?", "Envio de documento confidencial a servico externo sem garantias de sigilo.", ["observar a classificacao de confidencialidade", "usar ferramenta corporativa aprovada", "verificar termos de uso do servico de traducao"]),
        ("Intermediaria", "Durante uma reuniao com IA gerando atas automaticamente, sao discutidos numeros financeiros nao publicados do trimestre.", "Voce mantem a gravacao/transcricao por IA ativa normalmente?", "Registro e possivel retencao de dados financeiros sensiveis por terceiro (fornecedor de IA).", ["avaliar politica de retencao do fornecedor de IA", "considerar pausar a gravacao em trechos sensiveis", "verificar acordo de confidencialidade com o fornecedor"]),
        ("Avancada", "Sua equipe quer usar um modelo de IA de terceiros para analisar propostas de licitacao antes do prazo de submissao de concorrentes.", "Que riscos e cuidados voce aponta nesse uso?", "Possivel vazamento de estrategia competitiva confidencial e questoes concorrenciais/legais.", ["avaliar sensibilidade estrategica da informacao", "verificar isolamento e sigilo contratual do fornecedor", "considerar implicacoes juridicas e concorrenciais"]),
        ("Basica", "Um colega pede para voce colar o conteudo de um e-mail interno confidencial em uma IA para 'deixar o texto mais formal'.", "Voce atende ao pedido diretamente?", "Exposicao desnecessaria de comunicacao interna confidencial a servico externo.", ["questionar a real necessidade de usar IA externa", "remover trechos sensiveis antes de compartilhar", "sugerir ferramenta interna homologada"]),
        ("Intermediaria", "Voce recebeu acesso a um repositorio de documentos confidenciais de clientes e quer construir um assistente de busca com IA para facilitar consultas.", "Como voce projetaria esse assistente com seguranca?", "Exposicao ampla de documentos confidenciais de clientes por controle de acesso insuficiente.", ["implementar controle de acesso por permissao", "registrar auditoria de consultas", "restringir escopo de documentos indexados"]),
        ("Avancada", "Um ex-funcionario contata voce pedindo para 'so confirmar' informacoes de um projeto confidencial usando um resumo gerado por IA que ele diz ja ter.", "Como voce lida com essa situacao?", "Confirmacao inadvertida de informacoes confidenciais a pessoa sem autorizacao de acesso.", ["nao confirmar nem negar detalhes confidenciais", "verificar se a pessoa ainda tem autorizacao de acesso", "reportar a tentativa ao time de seguranca"]),
        ("Basica", "Voce quer usar uma IA para criar um resumo executivo de um relatorio interno que ainda nao foi aprovado pela diretoria.", "Voce compartilha o rascunho completo com a ferramenta de IA?", "Divulgacao prematura de informacao interna nao aprovada para publicacao/uso externo.", ["avaliar o status de aprovacao do documento", "usar apenas trechos ja autorizados", "verificar politica de divulgacao de rascunhos"]),
    ]

    engenharia_social = [
        ("Basica", "Voce recebe uma mensagem gerada aparentemente por um chatbot de suporte pedindo sua senha corporativa para 'verificar a conta'.", "Voce informa a senha?", "Comprometimento de credenciais corporativas via engenharia social.", ["reconhecer sinal de phishing/engenharia social", "nunca compartilhar senha", "reportar a tentativa ao time de seguranca"]),
        ("Intermediaria", "Um e-mail, aparentemente escrito com ajuda de IA, imita o estilo de escrita do seu diretor e pede uma transferencia urgente de dados financeiros.", "Como voce procede?", "Fraude do tipo 'CEO fraud' potencializada por IA generativa imitando estilo de escrita.", ["desconfiar de urgencia incomum", "verificar remetente por canal alternativo", "confirmar pedido antes de agir"]),
        ("Avancada", "Voce recebe uma ligacao com voz clonada por IA que soa identica a de um colega pedindo acesso emergencial a um sistema.", "Voce concede o acesso solicitado?", "Ataque de deepfake de voz para obter acesso indevido a sistemas internos.", ["desconfiar mesmo com voz reconhecivel", "validar por outro canal (mensagem, presencial)", "seguir protocolo formal de concessao de acesso"]),
        ("Basica", "Um perfil desconhecido em uma rede profissional, usando textos aparentemente gerados por IA, oferece uma vaga incrivel e pede seus dados bancarios 'para o contrato'.", "Voce responde com seus dados?", "Golpe de engenharia social explorando oferta de emprego falsa.", ["reconhecer sinais de golpe (pedido prematuro de dados)", "verificar autenticidade da empresa/recrutador", "evitar fornecer dados financeiros antecipadamente"]),
        ("Intermediaria", "Voce recebe um link para 'testar uma nova ferramenta de IA da empresa' enviado por um contato nao usual, com senso de urgencia.", "Voce clica no link imediatamente?", "Possivel link malicioso/phishing disfarçado de ferramenta de IA legitima.", ["verificar remetente e dominio do link", "confirmar com o time de TI antes de acessar", "nao clicar sob pressao de urgencia"]),
        ("Avancada", "Um atacante usa um assistente de IA para gerar dezenas de e-mails personalizados imitando fornecedores reais da empresa, pedindo atualizacao de dados bancarios.", "Como a empresa deveria se proteger desse tipo de ataque em escala?", "Fraude de fornecedor em escala usando IA para personalizacao de phishing.", ["implementar verificacao de mudanca de dados bancarios por canal separado", "treinar equipe para reconhecer padroes de phishing gerado por IA", "usar filtros e autenticacao de e-mail (SPF/DKIM/DMARC)"]),
        ("Basica", "Uma mensagem de texto, com linguagem muito natural (possivelmente gerada por IA), diz que sua conta sera bloqueada e pede que voce clique em um link agora.", "Voce clica no link para evitar o bloqueio?", "Smishing (phishing por SMS) usando urgencia artificial.", ["reconhecer padrao de urgencia suspeita", "nao clicar em links de mensagens nao solicitadas", "acessar o servico diretamente pelo site/app oficial"]),
        ("Intermediaria", "Um 'candidato a emprego' envia um curriculo em PDF que, segundo alertas do sistema, pode conter macros/scripts, pedindo que voce abra rapidamente pois a vaga fecha hoje.", "Voce abre o arquivo imediatamente?", "Possivel malware disfarçado de curriculo, com pressao de tempo como tatica de engenharia social.", ["desconfiar de pressao de tempo", "verificar o arquivo antes de abrir (antivirus/sandbox)", "seguir protocolo de seguranca para anexos externos"]),
        ("Avancada", "Voce recebe uma solicitacao interna, aparentemente automatizada por um assistente de IA de RH, pedindo que atualize dados bancarios de um funcionario diretamente em um link externo.", "Voce segue o link e atualiza os dados?", "Comprometimento de processo de RH por automacao maliciosa/phishing direcionado.", ["confirmar a solicitacao pelo canal oficial de RH", "desconfiar de automacoes que pedem dados sensiveis via link externo", "reportar a solicitacao suspeita"]),
        ("Basica", "Um colega te pede, por mensagem instantanea muito bem escrita (possivelmente por IA), para compartilhar seu login 'so por um minuto' para resolver um problema urgente.", "Voce compartilha seu login?", "Compartilhamento de credenciais pessoais facilitando acesso indevido.", ["nunca compartilhar credenciais pessoais", "oferecer ajuda por canal apropriado (chamado de TI)", "reportar pedidos incomuns de compartilhamento de senha"]),
    ]

    confiabilidade_ia = [
        ("Basica", "Uma IA generativa afirma com muita confianca um dado estatistico especifico sobre o mercado que voce nunca viu antes.", "Voce usa esse dado diretamente em um relatorio sem verificar?", "Uso de informacao potencialmente incorreta ('alucinacao') da IA sem verificacao.", ["questionar a confiabilidade da resposta", "buscar fonte primaria para confirmar o dado", "evitar usar informacao nao verificada em decisoes importantes"]),
        ("Intermediaria", "Voce pede a uma IA para resumir um artigo cientifico e ela cita um numero de pagina e uma citacao que parecem plausiveis.", "Voce confia na citacao sem checar o artigo original?", "Citacao inventada pela IA (alucinacao) usada como se fosse real.", ["verificar a fonte original antes de citar", "desconfiar de detalhes excessivamente especificos sem fonte", "reconhecer limitacoes de precisao factual da IA"]),
        ("Avancada", "Um modelo de IA usado para triagem de curriculos recomenda rejeitar candidatos de uma determinada regiao com uma taxa muito maior que a media.", "Como voce investigaria essa situacao?", "Vies algoritmico (bias) do modelo levando a decisoes potencialmente discriminatorias.", ["questionar possivel vies no modelo", "auditar os dados de treinamento e resultados por grupo", "envolver especialistas antes de usar o modelo em decisoes de impacto"]),
        ("Basica", "Uma IA sugere um medicamento e dosagem para um sintoma que voce descreveu informalmente.", "Voce segue a recomendacao sem consultar um profissional de saude?", "Uso de recomendacao de IA em contexto de saude sem validacao profissional, risco a seguranca fisica.", ["reconhecer limite de competencia da IA em saude", "buscar validacao com profissional qualificado", "nao tomar decisoes criticas de saude apenas com a IA"]),
        ("Intermediaria", "Voce usa uma IA para gerar codigo de um sistema financeiro e ela entrega uma solucao que parece funcionar nos testes rapidos.", "Voce coloca o codigo em producao sem revisao adicional?", "Falhas ocultas de logica ou seguranca em codigo gerado por IA sem revisao humana.", ["revisar o codigo gerado antes de usar", "testar cenarios de borda", "nao assumir corretude apenas por rodar sem erro"]),
        ("Avancada", "Uma IA usada para prever risco de credito apresenta alta acuracia geral, mas voce percebe desempenho bem pior para um subgrupo especifico de clientes.", "O que voce faria antes de aprovar o uso do modelo?", "Desempenho desigual do modelo (fairness) mascarado por metrica agregada de acuracia.", ["avaliar metricas por subgrupo, nao so agregadas", "investigar causa da disparidade", "considerar mitigacao de vies antes de implantar"]),
        ("Basica", "Uma IA generativa cria uma imagem 'realista' de um evento que voce sabe que nao aconteceu, e um colega quer compartilhar como se fosse real.", "Voce ajuda a compartilhar a imagem sem alertar sobre a origem?", "Disseminacao de desinformacao gerada por IA (imagem sintetica) como se fosse real.", ["identificar conteudo sintetico/gerado por IA", "alertar sobre a natureza da imagem antes de compartilhar", "evitar disseminar conteudo sem verificacao"]),
        ("Intermediaria", "Voce pede a uma IA uma analise juridica sobre um contrato e ela apresenta a resposta com tom muito confiante e definitivo.", "Voce trata a resposta como parecer juridico valido?", "Excesso de confianca do usuario em resposta de IA sem validacao por especialista, em area de alto risco.", ["reconhecer que confianca no tom nao implica correcao", "buscar validacao de um profissional da area", "usar a IA como apoio, nao como decisao final"]),
        ("Avancada", "Um sistema de IA usado para monitorar seguranca de uma planta industrial comeca a apresentar previsoes inconsistentes apos uma atualizacao.", "Que processo voce recomendaria para lidar com essa mudanca de comportamento?", "Degradacao de desempenho do modelo (model drift) nao detectada, com risco operacional.", ["monitorar continuamente o desempenho do modelo em producao", "estabelecer alertas para mudancas de comportamento", "ter plano de contingencia/rollback"]),
        ("Basica", "Uma IA erra o calculo de uma soma simples em um relatorio financeiro que voce esta revisando rapidamente.", "Voce assume que o calculo esta certo porque veio da IA?", "Confianca excessiva ('automation bias') levando a erro nao detectado.", ["conferir calculos criticos manualmente ou com outra ferramenta", "nao assumir corretude automatica de saidas de IA", "tratar a IA como apoio, com revisao humana"]),
    ]

    propriedade_intelectual = [
        ("Basica", "Voce quer usar uma IA generativa de imagens para criar a logo final da empresa, copiando o estilo de uma marca famosa.", "Voce pede explicitamente para 'imitar' a marca famosa?", "Possivel violacao de marca registrada/direitos autorais ao imitar identidade visual de terceiros.", ["reconhecer risco de violacao de propriedade intelectual", "evitar pedir imitacao direta de marcas/obras protegidas", "buscar um estilo original"]),
        ("Intermediaria", "Uma IA gera um texto para o site da empresa que, ao verificar, e quase identico a um trecho de um blog de terceiros.", "Voce publica o texto sem alteracoes?", "Publicacao de conteudo potencialmente plagiado gerado por IA, violando direitos autorais.", ["verificar originalidade do conteudo gerado", "reescrever trechos muito similares a fontes existentes", "citar fontes quando apropriado"]),
        ("Avancada", "Sua empresa quer treinar um modelo de IA proprio usando um grande volume de artigos e imagens coletados da internet sem verificar licencas.", "Que cuidados juridicos voce recomendaria antes de prosseguir?", "Uso nao autorizado de obras protegidas por direitos autorais no treinamento do modelo.", ["verificar licencas e termos de uso do conteudo", "considerar bases de dados com licenca adequada", "consultar time juridico sobre riscos de direitos autorais"]),
        ("Basica", "Voce usa uma IA para gerar musica de fundo para um video institucional e ela produz algo muito parecido com uma musica famosa conhecida.", "Voce usa a musica gerada sem verificar semelhanca?", "Possivel violacao de direitos autorais musicais por semelhanca substancial com obra existente.", ["comparar com obras conhecidas antes de usar", "buscar alternativas claramente originais ou licenciadas", "consultar orientacao sobre uso de musica gerada por IA"]),
        ("Intermediaria", "Um designer pede a uma IA para 'criar algo no estilo' de um ilustrador especifico e famoso para um material comercial.", "Isso representa um risco de propriedade intelectual?", "Uso comercial de estilo artistico especifico podendo gerar disputa sobre direitos morais/autorais do artista.", ["avaliar risco de associar a marca ao estilo de terceiro", "buscar autorizacao ou estilo suficientemente diferenciado", "considerar impacto reputacional e juridico"]),
        ("Avancada", "Voce descobre que um material de treinamento da empresa, criado com apoio de IA, reproduz quase integralmente um capitulo de um livro didatico protegido.", "Como voce conduziria a correcao dessa situacao?", "Reproducao substancial de obra protegida por direitos autorais sem autorizacao.", ["identificar a reproducao indevida", "remover ou reescrever o conteudo", "avaliar necessidade de licenciamento ou substituicao"]),
        ("Basica", "Voce pede a uma IA para escrever um resumo de um livro e pretende publicar o resumo como se fosse conteudo totalmente original seu.", "Isso e adequado sem nenhuma atribuicao?", "Falta de atribuicao adequada e possivel apropriacao indevida de conteudo derivado de obra protegida.", ["reconhecer a obra original como fonte", "dar credito adequado quando aplicavel", "evitar apresentar como 100% original algo derivado"]),
        ("Intermediaria", "Sua equipe de marketing usa uma IA para gerar variacoes de um slogan que acaba ficando muito parecido com o de um concorrente conhecido.", "Voce aprova o uso do slogan gerado?", "Risco de violacao de marca e concorrencia desleal por semelhanca com slogan de terceiro.", ["comparar com marcas e slogans existentes", "buscar assessoria juridica em caso de duvida", "optar por alternativa claramente distinta"]),
        ("Avancada", "Um pesquisador quer publicar um artigo cientifico usando trechos de texto gerados por IA sem indicar claramente o uso da ferramenta.", "Que orientacao voce daria sobre transparencia e autoria?", "Falta de transparencia sobre uso de IA em producao academica, com implicacoes eticas e de integridade.", ["declarar explicitamente o uso de ferramentas de IA", "verificar politicas da revista/instituicao sobre uso de IA", "garantir revisao e responsabilidade humana pelo conteudo final"]),
        ("Basica", "Voce encontrou uma imagem gerada por IA online que parece perfeita para uma apresentacao e quer usa-la sem verificar a licenca.", "Voce usa a imagem diretamente?", "Uso de conteudo gerado por IA com origem/licenca incerta, podendo violar termos de terceiros.", ["verificar a licenca e origem da imagem", "preferir bancos de imagens/IA com licenca clara para uso comercial", "documentar a fonte utilizada"]),
    ]

    def add_group(category, items):
        nonlocal sid
        for dificuldade, contexto, situacao, riscos, criterios in items:
            scenarios.append({
                "id": sid,
                "categoria": category,
                "dificuldade": dificuldade,
                "contexto": contexto,
                "situacao": situacao,
                "riscos": riscos,
                "criterios_avaliacao": criterios,
            })
            sid += 1

    add_group("Dados pessoais", dados_pessoais)
    add_group("Informacoes confidenciais", confidenciais)
    add_group("Engenharia social", engenharia_social)
    add_group("Confiabilidade e limitacoes da IA", confiabilidade_ia)
    add_group("Propriedade intelectual e direitos autorais", propriedade_intelectual)
    return scenarios


SCENARIOS = build_scenarios()
SCENARIOS_BY_ID = {s["id"]: s for s in SCENARIOS}

# ---------------------------------------------------------------------------
# Geracao do dataset sintetico (Secao 6 do notebook)
# ---------------------------------------------------------------------------


def _sample_user_skill_profile(rng):
    profile = {}
    for cat in CATEGORIES:
        a = rng.uniform(1.2, 3.0)
        b = rng.uniform(1.2, 3.0)
        profile[cat] = rng.beta(a, b)
    return profile


def generate_dataset(n_users=60, answers_per_user=15, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        skill = _sample_user_skill_profile(rng)
        general_skill = float(np.mean(list(skill.values())))
        chosen = rng.choice(SCENARIOS, size=answers_per_user, replace=True)
        for sc in chosen:
            cat = sc["categoria"]
            cat_skill = skill[cat]
            diff_penalty = {"Basica": 0.0, "Intermediaria": 0.08, "Avancada": 0.15}[sc["dificuldade"]]
            p_base = np.clip(0.55 * cat_skill + 0.45 * general_skill - diff_penalty, 0.03, 0.97)

            wmax = max(CATEGORY_FEATURE_WEIGHTS[cat].values())
            feat = {}
            for f in FEATURES:
                w = CATEGORY_FEATURE_WEIGHTS[cat][f]
                relevance = w / wmax
                p = np.clip(p_base * relevance + 0.5 * (1 - relevance), 0.03, 0.97)
                feat[f] = int(rng.random() < p)

            decision_score = float(np.mean([feat[f] * CATEGORY_FEATURE_WEIGHTS[cat][f] for f in FEATURES]) /
                                    np.mean(list(CATEGORY_FEATURE_WEIGHTS[cat].values())))
            decision_score = float(np.clip(decision_score, 0, 1))

            noise = rng.normal(0, 0.06)
            score_noisy = np.clip(decision_score + noise, 0, 1)
            if score_noisy >= 0.66:
                risk_level = "Baixo"
            elif score_noisy >= 0.38:
                risk_level = "Medio"
            else:
                risk_level = "Alto"

            rows.append({
                "user_id": u, "scenario_id": sc["id"], "category": cat,
                "difficulty": sc["dificuldade"], **feat,
                "decision_score": round(decision_score, 4), "risk_level": risk_level,
            })
    return pd.DataFrame(rows)


def _split_by_user(df, test_size=0.2, val_size=0.15, seed=RANDOM_SEED):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trainval, idx_test = next(gss1.split(df, groups=df["user_id"]))
    df_trainval, df_test = df.iloc[idx_trainval], df.iloc[idx_test]
    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    idx_train, idx_val = next(gss2.split(df_trainval, groups=df_trainval["user_id"]))
    return (df_trainval.iloc[idx_train].reset_index(drop=True),
            df_trainval.iloc[idx_val].reset_index(drop=True),
            df_test.reset_index(drop=True))


CAT_COLS = ["category", "difficulty"]
BIN_COLS = FEATURES


@dataclass
class TrainedEngine:
    preprocessor: ColumnTransformer
    model: object
    model_name: str
    label_encoder: LabelEncoder
    metrics: dict


def train_predictive_engine(seed=RANDOM_SEED):
    """Gera o dataset sintetico, treina e compara os 3 modelos e retorna o
    melhor deles (por F1 macro) ja encapsulado, pronto para prever."""
    dataset = generate_dataset(seed=seed)
    df_train, df_val, df_test = _split_by_user(dataset, seed=seed)

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("bin", StandardScaler(), BIN_COLS),
    ])
    X_train = df_train[CAT_COLS + BIN_COLS]
    y_train = df_train["risk_level"]
    X_test = df_test[CAT_COLS + BIN_COLS]
    y_test = df_test["risk_level"]

    preprocessor.fit(X_train)
    Xtr = preprocessor.transform(X_train)
    Xte = preprocessor.transform(X_test)

    label_encoder = LabelEncoder().fit(RISK_ORDER)

    candidates = {
        "Regressao Logistica": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(random_state=seed, eval_metric="mlogloss")
    else:
        candidates["Gradient Boosting"] = GradientBoostingClassifier(random_state=seed)

    all_metrics = {}
    fitted = {}
    for name, model in candidates.items():
        if name == "XGBoost":
            model.fit(Xtr, label_encoder.transform(y_train))
            pred = label_encoder.inverse_transform(model.predict(Xte))
        else:
            model.fit(Xtr, y_train)
            pred = model.predict(Xte)
        acc = accuracy_score(y_test, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, labels=RISK_ORDER, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=RISK_ORDER).tolist()
        all_metrics[name] = {"accuracy": round(float(acc), 4), "precision_macro": round(float(prec), 4),
                              "recall_macro": round(float(rec), 4), "f1_macro": round(float(f1), 4),
                              "confusion_matrix": cm}
        fitted[name] = model

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1_macro"])
    return TrainedEngine(
        preprocessor=preprocessor, model=fitted[best_name], model_name=best_name,
        label_encoder=label_encoder,
        metrics={"models": all_metrics, "selected_model": best_name,
                 "dataset_size": len(dataset), "note": "Metricas calculadas sobre dataset SINTETICO."},
    )


def predict_risk(engine: TrainedEngine, feat: dict, category: str, difficulty: str) -> str:
    linha = pd.DataFrame([{**feat, "category": category, "difficulty": difficulty}])
    X = linha[CAT_COLS + BIN_COLS]
    Xp = engine.preprocessor.transform(X)
    if engine.model_name == "XGBoost":
        pred_num = engine.model.predict(Xp)
        return str(engine.label_encoder.inverse_transform(pred_num)[0])
    return str(engine.model.predict(Xp)[0])


def compute_decision_score(feat: dict, category: str) -> float:
    weights = CATEGORY_FEATURE_WEIGHTS[category]
    wsum = sum(weights.values())
    return float(sum(feat[f] * weights[f] for f in FEATURES) / wsum)


# ---------------------------------------------------------------------------
# Perfil de vulnerabilidade / SSI / niveis (Secoes 11-13)
# ---------------------------------------------------------------------------


def compute_vulnerability_profile(history_rows):
    """history_rows: lista de dicts com pelo menos 'category' e as FEATURES.
    Retorna {categoria: score 0-100 (100 = mais vulneravel) ou None}."""
    df = pd.DataFrame(history_rows) if history_rows else pd.DataFrame(columns=["category"] + FEATURES)
    profile = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat] if len(df) else df
        if len(sub) == 0:
            profile[cat] = None
            continue
        weights = CATEGORY_FEATURE_WEIGHTS[cat]
        wsum = sum(weights.values())
        success = (sub[FEATURES] * pd.Series(weights)).sum(axis=1) / wsum
        profile[cat] = round(float(100 * (1 - success.mean())), 2)
    return profile


def compute_ssi(history_rows):
    if not history_rows:
        return None
    scores = [r["decision_score"] for r in history_rows]
    return round(float(100 * (sum(scores) / len(scores))), 2)


def determine_level(ssi, n_completed, min_scenarios=8):
    if ssi is None or n_completed < min_scenarios:
        return "Nivel 1 - Iniciante"
    if ssi < 40:
        return "Nivel 1 - Iniciante"
    elif ssi < 60:
        return "Nivel 2 - Consciente"
    elif ssi < 80:
        return "Nivel 3 - Preparado"
    return "Nivel 4 - Especialista"


# ---------------------------------------------------------------------------
# Motor de treinamento adaptativo (Secao 12/14)
# ---------------------------------------------------------------------------
ALPHA_BIAS = 1.6
EPSILON_EXPLORE = 0.10
TREND_WINDOW = 4


def choose_next_category(vulnerability_profile, rng):
    cats = list(vulnerability_profile.keys())
    vulns = np.array([vulnerability_profile[c] if vulnerability_profile[c] is not None else 50.0 for c in cats])
    weights = np.clip(vulns, 1, 100) ** ALPHA_BIAS
    weights = weights / weights.sum()
    if rng.random() < EPSILON_EXPLORE:
        return rng.choice(cats)
    return str(rng.choice(cats, p=weights))


def choose_next_difficulty(history_rows, category, current_difficulty, rng):
    sub = [r for r in history_rows if r["category"] == category][-TREND_WINDOW:]
    idx_cur = DIFFICULTY_ORDER.index(current_difficulty)
    if len(sub) < 2:
        return current_difficulty
    scores = [r["decision_score"] for r in sub]
    trend = scores[-1] - scores[0]
    mean_recent = sum(scores) / len(scores)
    if mean_recent >= 0.75 and idx_cur < len(DIFFICULTY_ORDER) - 1:
        return DIFFICULTY_ORDER[idx_cur + 1]
    if trend > 0.1 and idx_cur < len(DIFFICULTY_ORDER) - 1:
        return DIFFICULTY_ORDER[idx_cur + 1]
    if mean_recent < 0.35 and idx_cur > 0:
        return DIFFICULTY_ORDER[idx_cur - 1]
    return current_difficulty


def pick_scenario(category, difficulty, rng, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    pool = [s for s in SCENARIOS if s["categoria"] == category and s["dificuldade"] == difficulty
            and s["id"] not in exclude_ids]
    if not pool:
        pool = [s for s in SCENARIOS if s["categoria"] == category and s["id"] not in exclude_ids]
    if not pool:
        pool = [s for s in SCENARIOS if s["id"] not in exclude_ids]
    if not pool:
        pool = SCENARIOS
    idx = rng.integers(0, len(pool))
    return pool[idx]


def choose_next_scenario(history_rows, current_difficulty_by_cat):
    """Funcao de conveniencia usada pelo servidor: decide categoria,
    dificuldade e sorteia o cenario, tudo em uma chamada."""
    vuln_profile = compute_vulnerability_profile(history_rows)
    vuln_profile = {c: (50.0 if v is None else v) for c, v in vuln_profile.items()}
    rng = np.random.default_rng()

    category = choose_next_category(vuln_profile, rng)
    difficulty = choose_next_difficulty(history_rows, category, current_difficulty_by_cat.get(category, "Basica"), rng)
    used_ids = {r["scenario_id"] for r in history_rows}
    scenario = pick_scenario(category, difficulty, rng, exclude_ids=used_ids)
    return scenario, category, difficulty
