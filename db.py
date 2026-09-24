"""
Persistencia simples em SQLite (um unico arquivo skillshield.db).
Sem dependencias externas alem da biblioteca padrao do Python.
"""

import sqlite3
import os
from datetime import datetime, timezone

from ml_core import FEATURES

DB_PATH = os.environ.get("SKILLSHIELD_DB_PATH", os.path.join(os.path.dirname(__file__), "skillshield.db"))

_FEATURE_COLS_SQL = ", ".join(f"{f} INTEGER NOT NULL" for f in FEATURES)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            criado_em TEXT NOT NULL
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS respostas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            scenario_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            {_FEATURE_COLS_SQL},
            decision_score REAL NOT NULL,
            risk_level_previsto TEXT NOT NULL,
            fonte_resposta TEXT NOT NULL,
            resposta_texto TEXT,
            criado_em TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES usuarios(user_id)
        )
    """)
    conn.commit()
    conn.close()


def criar_usuario(nome: str) -> dict:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO usuarios (nome, criado_em) VALUES (?, ?)",
        (nome, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"user_id": user_id, "nome": nome}


def buscar_usuario(user_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM usuarios WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def salvar_resposta(user_id, scenario_id, category, difficulty, feat: dict,
                     decision_score, risk_level_previsto, fonte_resposta, resposta_texto=None):
    conn = get_conn()
    cols = ["user_id", "scenario_id", "category", "difficulty"] + FEATURES + [
        "decision_score", "risk_level_previsto", "fonte_resposta", "resposta_texto", "criado_em"]
    placeholders = ", ".join("?" for _ in cols)
    values = [user_id, scenario_id, category, difficulty] + [feat[f] for f in FEATURES] + [
        decision_score, risk_level_previsto, fonte_resposta, resposta_texto,
        datetime.now(timezone.utc).isoformat()]
    conn.execute(f"INSERT INTO respostas ({', '.join(cols)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


def historico_usuario(user_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM respostas WHERE user_id = ? ORDER BY id ASC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
