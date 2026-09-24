"""
Camada de identificacao via LLM (Google Gemini).

Recebe a resposta em texto livre do usuario para um cenario e devolve o
vetor de 9 caracteristicas booleanas (mesma estrutura usada pelo nucleo
preditivo em ml_core.py). Roda no servidor (nunca no navegador), para que
a chave de API do Google nunca seja exposta ao cliente.
"""

import json
import os

from ml_core import FEATURES, FEATURE_QUESTIONS

_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiNotConfigured(RuntimeError):
    pass


def _get_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiNotConfigured(
            "GOOGLE_API_KEY nao configurada. Defina a variavel de ambiente "
            "(veja o .env.example) com uma chave criada em "
            "https://aistudio.google.com/apikey"
        )
    from google import genai  # import tardio: so falha se realmente for chamado sem a lib instalada
    return genai.Client(api_key=api_key)


def _montar_prompt(scenario: dict, resposta_livre: str) -> str:
    perguntas_txt = "\n".join(f"- {k}: {v}" for k, v in FEATURE_QUESTIONS.items())
    criterios_txt = "\n".join(f"- {c}" for c in scenario.get("criterios_avaliacao", []))
    return f"""Voce e um avaliador de comportamento de seguranca no uso de IA no trabalho.

CENARIO (categoria: {scenario['categoria']}, dificuldade: {scenario['dificuldade']}):
{scenario['contexto']}
{scenario['situacao']}

Criterios de avaliacao esperados para este cenario:
{criterios_txt}

RESPOSTA DO USUARIO (texto livre, o que ele disse que faria):
\"\"\"{resposta_livre}\"\"\"

Para cada um dos comportamentos abaixo, decida se a resposta do usuario demonstra esse comportamento (true) ou nao (false):
{perguntas_txt}

Responda apenas com um objeto JSON contendo exatamente essas 9 chaves com valores true/false, sem nenhum texto adicional."""


def extract_features(scenario: dict, resposta_livre: str) -> dict:
    """Chama o Gemini e retorna {feature: 0/1}. Em caso de erro (rede, chave
    invalida, resposta malformada), levanta a excecao para o chamador decidir
    como avisar o usuario (o app.py trata isso e nunca inventa um resultado
    silenciosamente)."""
    from google.genai import types

    client = _get_client()
    prompt = _montar_prompt(scenario, resposta_livre)

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={f: types.Schema(type=types.Type.BOOLEAN) for f in FEATURES},
        required=FEATURES,
    )

    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    dados = json.loads(response.text)
    return {f: int(bool(dados.get(f, False))) for f in FEATURES}
