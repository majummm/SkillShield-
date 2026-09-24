const FEATURE_LABELS = {
  identified_risk: "Percebeu o risco da situação",
  protected_data: "Evitou expor dados pessoais",
  avoided_sharing: "Evitou compartilhar sem cuidado",
  used_anonymization: "Considerou anonimizar/mascarar",
  checked_policy: "Verificaria a política da empresa",
  verified_information: "Verificaria a fonte/confiabilidade",
  recognized_social_engineering: "Reconheceu sinais de golpe",
  questioned_ai: "Questionaria a resposta da IA",
  considered_intellectual_property: "Considerou direitos autorais",
};

const RISK_LABEL = { Baixo: "BAIXO RISCO", Medio: "MÉDIO RISCO", Alto: "ALTO RISCO" };
const RISK_CLASS = { Baixo: "risco-baixo", Medio: "risco-medio", Alto: "risco-alto" };

const state = {
  userId: Number(localStorage.getItem("skillshield_user_id")) || null,
  nome: localStorage.getItem("skillshield_nome") || null,
  cenarioAtual: null,
};

const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.erro || `Erro ${res.status}`);
  }
  return data;
}

function corVulnerabilidade(valor) {
  // 0 (pouco vulneravel) -> verde ; 100 (muito vulneravel) -> vermelho
  if (valor === null || valor === undefined) return "#DCD2B8";
  if (valor < 34) return "#3F6F52";
  if (valor < 67) return "#B4741E";
  return "#9C3B3B";
}

function renderPerfil(perfil) {
  el("ssi-valor").textContent = perfil.ssi === null ? "—" : Math.round(perfil.ssi);
  const pct = perfil.ssi === null ? 0 : Math.max(0, Math.min(100, perfil.ssi));
  el("ssi-anel").style.background =
    `conic-gradient(var(--ink) ${pct * 3.6}deg, var(--line) ${pct * 3.6}deg)`;
  el("dossie-nivel-texto").textContent = perfil.nivel.replace(" - ", " — ");
  el("dossie-contagem").textContent =
    `${perfil.cenarios_respondidos} caso${perfil.cenarios_respondidos === 1 ? "" : "s"} respondido${perfil.cenarios_respondidos === 1 ? "" : "s"}`;

  const cont = el("vuln-barras");
  cont.innerHTML = "";
  Object.entries(perfil.perfil_vulnerabilidade).forEach(([categoria, valor]) => {
    const linha = document.createElement("div");
    linha.className = "vuln-linha";
    const label = document.createElement("div");
    label.className = "vuln-linha-label";
    label.innerHTML = `<span>${categoria}</span><span class="vuln-linha-valor">${valor === null ? "—" : Math.round(valor)}</span>`;
    const fundo = document.createElement("div");
    fundo.className = "vuln-barra-fundo";
    const preenchimento = document.createElement("div");
    preenchimento.className = "vuln-barra-preenchimento";
    preenchimento.style.width = (valor === null ? 0 : valor) + "%";
    preenchimento.style.background = corVulnerabilidade(valor);
    fundo.appendChild(preenchimento);
    linha.appendChild(label);
    linha.appendChild(fundo);
    cont.appendChild(linha);
  });
}

async function carregarPerfil() {
  const perfil = await api(`/api/perfil/${state.userId}`);
  renderPerfil(perfil);
}

async function carregarProximoCenario() {
  el("veredito").classList.add("hidden");
  el("caso-card").classList.remove("hidden");
  el("resposta-livre").value = "";
  el("envio-erro").classList.add("hidden");
  el("btn-enviar").disabled = false;

  const { cenario } = await api(`/api/cenario/${state.userId}`);
  state.cenarioAtual = cenario;
  el("caso-id").textContent = cenario.id;
  el("caso-categoria").textContent = cenario.categoria;
  el("caso-dificuldade").textContent = cenario.dificuldade;
  el("caso-contexto").textContent = cenario.contexto;
  el("caso-situacao").textContent = cenario.situacao;
}

function renderVeredito(resultado) {
  el("caso-card").classList.add("hidden");
  const veredito = el("veredito");
  veredito.classList.remove("hidden");

  const stamp = el("stamp");
  stamp.textContent = RISK_LABEL[resultado.risco_previsto] || resultado.risco_previsto;
  stamp.className = "stamp " + (RISK_CLASS[resultado.risco_previsto] || "");

  const featuresUl = el("veredito-features");
  featuresUl.innerHTML = "";
  Object.entries(resultado.caracteristicas_identificadas).forEach(([chave, valor]) => {
    const li = document.createElement("li");
    li.textContent = FEATURE_LABELS[chave] || chave;
    li.className = valor ? "demonstrado" : "nao-demonstrado";
    li.textContent += valor ? " — identificado" : " — não identificado";
    featuresUl.appendChild(li);
  });

  const riscosUl = el("veredito-riscos");
  riscosUl.innerHTML = "";
  const riscos = Array.isArray(resultado.riscos_do_cenario) ? resultado.riscos_do_cenario : [resultado.riscos_do_cenario];
  riscos.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    riscosUl.appendChild(li);
  });

  renderPerfil(resultado.perfil);
}

async function irParaTelaPrincipal() {
  try {
    await carregarPerfil(); // valida que o usuario ainda existe no servidor
  } catch (e) {
    // usuario nao encontrado (ex.: banco de dados foi reiniciado) -> pede novo cadastro
    localStorage.removeItem("skillshield_user_id");
    localStorage.removeItem("skillshield_nome");
    state.userId = null;
    state.nome = null;
    return;
  }
  el("tela-cadastro").classList.add("hidden");
  el("tela-principal").classList.remove("hidden");
  el("dossie-numero").textContent = state.userId;
  el("dossie-nome").textContent = state.nome;
  carregarProximoCenario().catch(mostrarErroCarregarCenario);
}

function mostrarErroCarregarCenario(e) {
  console.error(e);
  el("caso-situacao").textContent = "Não foi possível carregar o próximo caso. Recarregue a página.";
}

el("btn-cadastrar").addEventListener("click", async () => {
  const nome = el("input-nome").value.trim();
  el("cadastro-erro").classList.add("hidden");
  if (!nome) {
    el("cadastro-erro").textContent = "Digite um nome para continuar.";
    el("cadastro-erro").classList.remove("hidden");
    return;
  }
  el("btn-cadastrar").disabled = true;
  try {
    const usuario = await api("/api/cadastro", { method: "POST", body: JSON.stringify({ nome }) });
    state.userId = usuario.user_id;
    state.nome = usuario.nome;
    localStorage.setItem("skillshield_user_id", usuario.user_id);
    localStorage.setItem("skillshield_nome", usuario.nome);
    irParaTelaPrincipal();
  } catch (e) {
    el("cadastro-erro").textContent = e.message;
    el("cadastro-erro").classList.remove("hidden");
  } finally {
    el("btn-cadastrar").disabled = false;
  }
});

el("btn-enviar").addEventListener("click", async () => {
  const resposta_livre = el("resposta-livre").value.trim();
  el("envio-erro").classList.add("hidden");
  if (!resposta_livre) {
    el("envio-erro").textContent = "Descreva o que você faria antes de enviar.";
    el("envio-erro").classList.remove("hidden");
    return;
  }
  el("btn-enviar").disabled = true;
  el("envio-carregando").classList.remove("hidden");
  try {
    const resultado = await api("/api/responder", {
      method: "POST",
      body: JSON.stringify({
        user_id: state.userId,
        scenario_id: state.cenarioAtual.id,
        resposta_livre,
      }),
    });
    renderVeredito(resultado);
  } catch (e) {
    el("envio-erro").textContent = e.message;
    el("envio-erro").classList.remove("hidden");
  } finally {
    el("btn-enviar").disabled = false;
    el("envio-carregando").classList.add("hidden");
  }
});

el("btn-proximo").addEventListener("click", () => {
  carregarProximoCenario().catch(mostrarErroCarregarCenario);
});

// Retoma a sessao automaticamente se o usuario ja tinha se cadastrado neste navegador
if (state.userId && state.nome) {
  irParaTelaPrincipal();
}
