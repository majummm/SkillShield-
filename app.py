"""
Servidor do SkillShield.

Roda o site inteiro: serve o front-end (templates/index.html + static/) e
expoe a API usada pelo JavaScript da pagina. A chave do Google Gemini fica
SOMENTE no servidor (variavel de ambiente GOOGLE_API_KEY) — nunca e enviada
ao navegador.

Para rodar localmente:
    pip install -r requirements.txt
    cp .env.example .env      # depois edite o .env com sua chave
    python app.py
    # abra http://localhost:5000
"""

import os

from flask import Flask, jsonify, request, render_template

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv e opcional; sem ele, use variaveis de ambiente do sistema

import db
import ml_core
from llm_gemini import extract_features, GeminiNotConfigured

app = Flask(__name__)

# Treina o motor preditivo uma vez, na subida do servidor (dataset sintetico,
# poucos segundos). Veja ml_core.py para a metodologia completa.
print("Treinando o nucleo preditivo do SkillShield (dados sinteticos)...")
ENGINE = ml_core.train_predictive_engine()
print(f"Modelo selecionado: {ENGINE.model_name} "
      f"(F1 macro = {ENGINE.metrics['models'][ENGINE.model_name]['f1_macro']})")

db.init_db()


def _scenario_publico(scenario: dict) -> dict:
    """Versao do cenario enviada ao cliente ANTES da resposta: nao inclui os
    criterios de avaliacao, para nao 'entregar a resposta' de antemao."""
    return {
        "id": scenario["id"],
        "categoria": scenario["categoria"],
        "dificuldade": scenario["dificuldade"],
        "contexto": scenario["contexto"],
        "situacao": scenario["situacao"],
    }


def _perfil_payload(user_id: int) -> dict:
    hist = db.historico_usuario(user_id)
    ssi = ml_core.compute_ssi(hist)
    perfil = ml_core.compute_vulnerability_profile(hist)
    nivel = ml_core.determine_level(ssi, len(hist))
    return {
        "cenarios_respondidos": len(hist),
        "ssi": ssi,
        "nivel": nivel,
        "perfil_vulnerabilidade": perfil,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "modelo_selecionado": ENGINE.model_name,
        "metricas": ENGINE.metrics,
        "gemini_configurado": bool(os.environ.get("GOOGLE_API_KEY")),
    })


@app.route("/api/cadastro", methods=["POST"])
def cadastro():
    data = request.get_json(force=True) or {}
    nome = (data.get("nome") or "").strip()
    if not nome:
        return jsonify({"erro": "Informe um nome."}), 400
    usuario = db.criar_usuario(nome)
    return jsonify(usuario)


@app.route("/api/cenario/<int:user_id>")
def proximo_cenario(user_id):
    if not db.buscar_usuario(user_id):
        return jsonify({"erro": "Usuario nao encontrado."}), 404
    hist = db.historico_usuario(user_id)
    current_difficulty = {}
    for r in hist:
        current_difficulty[r["category"]] = r["difficulty"]
    scenario, category, difficulty = ml_core.choose_next_scenario(hist, current_difficulty)
    return jsonify({"cenario": _scenario_publico(scenario)})


@app.route("/api/responder", methods=["POST"])
def responder():
    data = request.get_json(force=True) or {}
    user_id = data.get("user_id")
    scenario_id = data.get("scenario_id")
    resposta_livre = (data.get("resposta_livre") or "").strip()

    if not user_id or not db.buscar_usuario(user_id):
        return jsonify({"erro": "Usuario nao encontrado."}), 404
    scenario = ml_core.SCENARIOS_BY_ID.get(scenario_id)
    if not scenario:
        return jsonify({"erro": "Cenario nao encontrado."}), 404
    if not resposta_livre:
        return jsonify({"erro": "Descreva o que voce faria antes de enviar."}), 400

    try:
        feat = extract_features(scenario, resposta_livre)
    except GeminiNotConfigured as e:
        return jsonify({"erro": str(e)}), 503
    except Exception as e:
        return jsonify({"erro": f"Nao foi possivel interpretar a resposta com a IA do Google agora ({e}). "
                                 f"Tente novamente em instantes."}), 502

    category, difficulty = scenario["categoria"], scenario["dificuldade"]
    decision_score = ml_core.compute_decision_score(feat, category)
    risco_previsto = ml_core.predict_risk(ENGINE, feat, category, difficulty)

    db.salvar_resposta(
        user_id=user_id, scenario_id=scenario_id, category=category, difficulty=difficulty,
        feat=feat, decision_score=decision_score, risk_level_previsto=risco_previsto,
        fonte_resposta="gemini", resposta_texto=resposta_livre,
    )

    return jsonify({
        "caracteristicas_identificadas": feat,
        "decision_score": round(decision_score, 4),
        "risco_previsto": risco_previsto,
        "riscos_do_cenario": scenario["riscos"],
        "perfil": _perfil_payload(user_id),
    })


@app.route("/api/perfil/<int:user_id>")
def perfil(user_id):
    if not db.buscar_usuario(user_id):
        return jsonify({"erro": "Usuario nao encontrado."}), 404
    return jsonify(_perfil_payload(user_id))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
