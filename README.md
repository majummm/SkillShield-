# SkillShield — Site

Site completo do SkillShield: cadastro de usuário, apresentação de cenários
de risco pelo motor adaptativo, resposta em **texto livre** interpretada
pela **IA do Google (Gemini)**, e predição do nível de risco por um modelo
de Machine Learning treinado no próprio servidor.

```
Navegador (front-end)  --texto livre-->  Servidor Flask
                                             |
                                             |-- Gemini (Google) extrai as 9 caracteristicas
                                             |-- modelo de ML treinado preve o risco
                                             |-- SQLite guarda historico, perfil e SSI
                                             v
Navegador (front-end)  <--risco, perfil, SSI--
```

A chave de API do Google **fica só no servidor** (variável de ambiente) —
o navegador nunca tem acesso a ela.

## 1. Pré-requisitos

* Python 3.10 ou mais recente
* Uma chave de API gratuita do Google Gemini: https://aistudio.google.com/apikey

## 2. Instalação (rodando localmente)

```bash
cd skillshield_site
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edite o arquivo .env e cole sua GOOGLE_API_KEY

python app.py
```

Abra **http://localhost:5000** no navegador. Pronto — o site já está no ar
localmente, com o modelo de ML treinado automaticamente na inicialização
(leva poucos segundos, usando o dataset sintético descrito abaixo).

## 3. Como funciona por dentro

* `ml_core.py` — banco de 50 cenários, geração do dataset sintético de
  treinamento, treinamento/comparação dos modelos (Regressão Logística,
  Random Forest, Gradient Boosting/XGBoost), Perfil de Vulnerabilidade, SSI,
  níveis e motor de treinamento adaptativo.
* `llm_gemini.py` — chama o Gemini para transformar a resposta em texto
  livre no vetor de 9 características, usando `response_schema` para forçar
  saída em JSON estruturado.
* `db.py` — persistência simples em SQLite (`skillshield.db`, criado
  automaticamente na primeira execução).
* `app.py` — servidor Flask: serve a página e expõe a API (`/api/cadastro`,
  `/api/cenario/<id>`, `/api/responder`, `/api/perfil/<id>`).
* `templates/` e `static/` — front-end (HTML/CSS/JS puro, sem framework).

## 4. Importante — transparência científica

O modelo preditivo é treinado, a cada início do servidor, com um **dataset
sintético** gerado por regras (não há respostas de usuários reais nesses
dados de treino). Veja `ml_core.generate_dataset()` para a metodologia
completa. As métricas do modelo ficam disponíveis em `GET /api/status`.

Assim que houver volume suficiente de respostas reais (tabela `respostas`
do `skillshield.db`), o ideal é:
1. Definir com a equipe como obter um "risco verdadeiro" confiável para
   essas respostas (ex.: avaliação humana de uma amostra).
2. Substituir `generate_dataset()` por essas respostas reais rotuladas.
3. Retreinar e comparar os modelos novamente antes de confiar nas previsões
   em produção.

## 5. Colocando o site no ar (deploy)

Qualquer serviço que rode aplicações Python/Flask funciona. Alguns
caminhos comuns (todos têm planos gratuitos ou de baixo custo):

* **Render** (render.com) — conecte o repositório, defina o *Start Command*
  como `python app.py` (ou `gunicorn app:app`), e cadastre `GOOGLE_API_KEY`
  em "Environment".
* **Railway** (railway.app) — similar ao Render, com deploy por Git.
* **PythonAnywhere** — bom para quem prefere subir os arquivos manualmente.
* **Google Cloud Run** — se quiser manter tudo dentro do ecossistema Google.

Em qualquer um deles, os passos são os mesmos:
1. Suba os arquivos deste projeto (sem o `.env`!).
2. Configure a variável de ambiente `GOOGLE_API_KEY` no painel do serviço.
3. Garanta que o disco onde fica `skillshield.db` seja persistente (em
   alguns serviços gratuitos o disco é temporário e reseta a cada deploy —
   nesse caso, considere trocar o SQLite por um banco gerenciado, como
   Postgres, se for usar o site com um número real de participantes).

## 6. Rodando em modo debug (opcional, só em desenvolvimento)

```bash
FLASK_DEBUG=1 python app.py
```

Nunca use `FLASK_DEBUG=1` em produção — ele expõe informações internas do
servidor a qualquer visitante.
