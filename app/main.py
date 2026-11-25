import logging
import time
import re
import os
import secrets
import threading
from collections import deque
from time import monotonic

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from pathlib import Path

import requests
import psycopg
from psycopg.types.json import Jsonb

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.teammate_context import (
    build_candidate_teammate_context,
    extract_team_player_status,
)
from app.availability_context import (
    parse_espn_injuries,
    select_eligible_players,
)
from app.nba_injury_reports import get_official_nba_injury_status
from app.audit_metrics import build_context_reference_metrics, build_market_score_bands, build_teammate_split_metrics, build_gap_group_metrics, build_model_comparison, build_top10_vs_rest_metrics, is_temporally_valid_peer, spearman_correlation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nba_props_scanner")

app = FastAPI(
    title="NBA Props Scanner",
    description="API de triagem estatística NBA via ESPN.",
    version="3.17.0"
)

# Rate limit simples e local para a camada HTTP.
_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))
_RATE_LIMIT_EXEMPT_PATHS = {"/health"}
_rate_limit_lock = threading.Lock()
_rate_limit_buckets: dict[str, deque[float]] = {}

def _rate_limit_key(request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"

def _is_admin_request(request) -> bool:
    configured_key = os.getenv("ADMIN_API_KEY")
    supplied_key = request.headers.get("X-Admin-Key")
    return bool(configured_key and supplied_key and secrets.compare_digest(supplied_key, configured_key))

@app.middleware("http")
async def rate_limit_requests(request, call_next):
    # O limite público protege somente a rota de geração de candidatos.
    # As telas fazem várias leituras independentes de auditoria/contexto; limitar
    # todas as rotas por IP bloqueava consultas legítimas e causava falhas visíveis.
    if request.url.path != "/candidates" or _is_admin_request(request):
        return await call_next(request)
    now = monotonic()
    cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
    key = _rate_limit_key(request)
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(key, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX_REQUESTS:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Limite de requisições excedido. Tente novamente em instantes."},
                headers={"Retry-After": str(_RATE_LIMIT_WINDOW_SECONDS)},
            )
        bucket.append(now)
    return await call_next(request)

# CORS restrito por ambiente. O frontend atual é servido pela própria aplicação;
# origens adicionais só entram mediante variável explícita.
_DEFAULT_CORS_ORIGINS = (
    "https://nba-props-scanner.onrender.com,"
    "https://nba-props-scanner-staging.onrender.com"
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Admin-Key"],
)


# ============================================================
# BANCO DE DADOS - HISTÓRICO IMUTÁVEL DE CANDIDATOS
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

# Chave administrativa para rotinas que alteram o histórico persistido.
# A chave nunca fica no código nem é registrada em logs.
admin_api_key_header = APIKeyHeader(
    name="X-Admin-Key",
    auto_error=False,
    description="Chave administrativa para rotinas internas de captura/coleta."
)


def require_admin_api_key(api_key: Optional[str] = Depends(admin_api_key_header)):
    """Protege rotinas internas de escrita/atualização com uma chave de serviço.

    Falha fechado quando a variável não existe: uma rota protegida não pode
    permanecer publicamente gravável por configuração incompleta.
    """
    configured_key = os.getenv("ADMIN_API_KEY")
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="Autenticação administrativa não configurada."
        )
    if not api_key or not secrets.compare_digest(api_key, configured_key):
        raise HTTPException(
            status_code=401,
            detail="Chave administrativa inválida ou ausente."
        )
    return True

AUDIT_TIMEZONE = ZoneInfo("America/Sao_Paulo")
AUDIT_START_DATE = date(2026, 11, 16)
# Coorte limpa para métricas acumuladas; preserva coleta e snapshots brutos.
AUDIT_METRICS_START_DATE = date(2026, 11, 19)


def audit_history_cutoff_date():
    """Data final da visão acumulada: sempre inclui o que já foi capturado até hoje."""
    return datetime.now(AUDIT_TIMEZONE).date()


def audit_history_start_date():
    """Primeiro dia oficial da auditoria independente e da coleta histórica."""
    return AUDIT_START_DATE


def audit_metrics_start_date():
    """Primeiro dia da coorte usada nos indicadores acumulados."""
    return AUDIT_METRICS_START_DATE




def get_database_connection():
    """Abre conexão com PostgreSQL/Neon sem expor credenciais."""

    if not DATABASE_URL:
        raise RuntimeError("Variável DATABASE_URL não configurada.")

    return psycopg.connect(
        DATABASE_URL,
        connect_timeout=10
    )


def initialize_database():
    """Cria a estrutura de histórico caso ainda não exista."""

    if not DATABASE_URL:
        logger.warning(
            "DATABASE_URL não configurada. Histórico ficará desativado."
        )
        return False

    try:
        with get_database_connection() as connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candidate_snapshots (
                        id BIGSERIAL PRIMARY KEY,
                        selected_date DATE NOT NULL,
                        market TEXT NOT NULL,
                        player_id TEXT NOT NULL,
                        player_name TEXT NOT NULL,
                        team TEXT,
                        opponent TEXT,
                        position TEXT,
                        triage_score DOUBLE PRECISION,
                        last5_avg DOUBLE PRECISION,
                        last10_avg DOUBLE PRECISION,
                        minutes_last5_avg DOUBLE PRECISION,
                        display_rank INTEGER,
                        payload JSONB NOT NULL,
                        captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT uq_candidate_snapshot
                            UNIQUE (selected_date, market, player_id)
                    )
                    """
                )

                cursor.execute(
                    """
                    ALTER TABLE candidate_snapshots
                    ADD COLUMN IF NOT EXISTS display_rank INTEGER
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_candidate_snapshots_date_market
                    ON candidate_snapshots (selected_date, market)
                    """
                )

                # População independente para auditoria: não representa uso do Scanner.
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_snapshots (
                        id BIGSERIAL PRIMARY KEY,
                        selected_date DATE NOT NULL,
                        market TEXT NOT NULL,
                        player_id TEXT NOT NULL,
                        player_name TEXT NOT NULL,
                        team TEXT,
                        opponent TEXT,
                        position TEXT,
                        triage_score DOUBLE PRECISION,
                        last5_avg DOUBLE PRECISION,
                        last10_avg DOUBLE PRECISION,
                        minutes_last5_avg DOUBLE PRECISION,
                        audit_rank INTEGER NOT NULL,
                        availability_status TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT uq_audit_snapshot
                            UNIQUE (selected_date, market, player_id)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_snapshots_date_market
                    ON audit_snapshots (selected_date, market)
                    """
                )

                # Manifesto também registra capturas válidas com zero elegíveis.
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_snapshot_runs (
                        id BIGSERIAL PRIMARY KEY,
                        selected_date DATE NOT NULL,
                        market TEXT NOT NULL,
                        eligible_count INTEGER NOT NULL,
                        availability_report_status TEXT NOT NULL,
                        captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT uq_audit_snapshot_run
                            UNIQUE (selected_date, market)
                    )
                    """
                )

                # Compatibilidade para snapshots independentes criados antes do manifesto.
                cursor.execute(
                    """
                    INSERT INTO audit_snapshot_runs (
                        selected_date, market, eligible_count,
                        availability_report_status
                    )
                    SELECT
                        selected_date, market, COUNT(*), 'legacy_existing'
                    FROM audit_snapshots
                    GROUP BY selected_date, market
                    ON CONFLICT (selected_date, market) DO NOTHING
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candidate_results (
                        id BIGSERIAL PRIMARY KEY,
                        selected_date DATE NOT NULL,
                        market TEXT NOT NULL,
                        player_id TEXT NOT NULL,
                        player_name TEXT NOT NULL,
                        team TEXT,
                        opponent TEXT,
                        actual_value DOUBLE PRECISION NOT NULL,
                        game_id TEXT,
                        collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB,
                        CONSTRAINT uq_candidate_result
                            UNIQUE (selected_date, market, player_id)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_candidate_results_date_market
                    ON candidate_results (selected_date, market)
                    """
                )

                # Resultados da população independente de auditoria.
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_results (
                        id BIGSERIAL PRIMARY KEY,
                        selected_date DATE NOT NULL,
                        market TEXT NOT NULL,
                        player_id TEXT NOT NULL,
                        player_name TEXT NOT NULL,
                        team TEXT,
                        opponent TEXT,
                        actual_value DOUBLE PRECISION NOT NULL,
                        game_id TEXT,
                        collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB,
                        CONSTRAINT uq_audit_result
                            UNIQUE (selected_date, market, player_id)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_results_date_market
                    ON audit_results (selected_date, market)
                    """
                )

                # Casos encerrados sem resultado estatístico, por exemplo DNP.
                # Ficam separados de audit_results para não contaminar métricas.
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_no_results (
                        id BIGSERIAL PRIMARY KEY,
                        selected_date DATE NOT NULL,
                        market TEXT NOT NULL,
                        player_id TEXT NOT NULL,
                        player_name TEXT NOT NULL,
                        team TEXT,
                        opponent TEXT,
                        reason TEXT NOT NULL,
                        game_id TEXT,
                        recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB,
                        CONSTRAINT uq_audit_no_result
                            UNIQUE (selected_date, market, player_id)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_no_results_date_market
                    ON audit_no_results (selected_date, market)
                    """
                )

                # Motivos persistidos para pendências técnicas da auditoria.
                # Não encerra a pendência; apenas registra por que ainda não foi resolvida.
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_pending_reasons (
                        id BIGSERIAL PRIMARY KEY,
                        selected_date DATE NOT NULL,
                        market TEXT NOT NULL,
                        player_id TEXT NOT NULL,
                        reason_code TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        detail TEXT,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT uq_audit_pending_reason
                            UNIQUE (selected_date, market, player_id)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_pending_reasons_date_market
                    ON audit_pending_reasons (selected_date, market)
                    """
                )

        logger.info(
            "Banco Neon inicializado. Tabela candidate_snapshots disponível."
        )

        return True

    except Exception as error:
        logger.exception(
            "Não foi possível inicializar o banco Neon: %s",
            error
        )
        return False


def save_candidate_snapshots(
    selected_date,
    market,
    candidates
):
    """
    Salva todos os candidatos antes do limite de exibição,
    sem sobrescrever a fotografia original.
    """

    if not candidates:
        return 0

    if not DATABASE_URL:
        logger.warning(
            "Histórico não salvo: DATABASE_URL não configurada."
        )
        return 0

    rows = []

    for display_rank, candidate in enumerate(candidates, start=1):

        player_id = candidate.get("player_id")

        if not player_id:
            logger.warning(
                "Candidato sem player_id ignorado no histórico: %s",
                candidate.get("player")
            )
            continue

        rows.append(
            (
                selected_date,
                market,
                str(player_id),
                candidate.get("player", ""),
                candidate.get("team"),
                candidate.get("opponent"),
                candidate.get("position"),
                candidate.get("triage_score"),
                candidate.get("last5_avg"),
                candidate.get("last10_avg"),
                candidate.get("minutes_last5_avg"),
                display_rank,
                Jsonb(candidate)
            )
        )

    if not rows:
        return 0

    try:

        with get_database_connection() as connection:

            with connection.cursor() as cursor:

                cursor.executemany(
                    """
                    INSERT INTO candidate_snapshots (
                        selected_date,
                        market,
                        player_id,
                        player_name,
                        team,
                        opponent,
                        position,
                        triage_score,
                        last5_avg,
                        last10_avg,
                        minutes_last5_avg,
                        display_rank,
                        payload
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (
                        selected_date,                        market,                        player_id
                    ) DO NOTHING
                    """,
                    rows
                )

        logger.info(
            "Histórico processado: %s candidatos enviados para %s/%s.",
            len(rows),
            selected_date,
            market
        )

        return len(rows)

    except Exception as error:

        logger.exception(
            "Falha ao salvar histórico de candidatos: %s",
            error
        )

        return 0


def save_audit_snapshots(
    selected_date,
    market,
    candidates,
    availability_report_status="unknown"
):
    """
    Persiste a população completa elegível para auditoria, separada do
    histórico operacional do Scanner. A primeira fotografia é imutável.
    """
    if not DATABASE_URL:
        logger.warning(
            "População de auditoria não salva: DATABASE_URL não configurada."
        )
        return 0

    rows = []
    for audit_rank, candidate in enumerate(candidates, start=1):
        player_id = candidate.get("player_id")
        if not player_id:
            logger.warning(
                "Jogador sem player_id ignorado no snapshot de auditoria: %s",
                candidate.get("player")
            )
            continue

        availability = candidate.get("availability") or {}
        availability_status = str(
            availability.get("availability") or "unknown"
        )
        rows.append((
            selected_date,
            market,
            str(player_id),
            candidate.get("player", ""),
            candidate.get("team"),
            candidate.get("opponent"),
            candidate.get("position"),
            candidate.get("triage_score"),
            candidate.get("last5_avg"),
            candidate.get("last10_avg"),
            candidate.get("minutes_last5_avg"),
            audit_rank,
            availability_status,
            Jsonb(candidate)
        ))

    try:
        with get_database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO audit_snapshot_runs (
                        selected_date,
                        market,
                        eligible_count,
                        availability_report_status
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (selected_date, market) DO NOTHING
                    RETURNING id
                    """,
                    (
                        selected_date,
                        market,
                        len(rows),
                        availability_report_status
                    )
                )
                if not cursor.fetchone():
                    logger.info(
                        "Captura de auditoria já registrada; mantendo primeira versão: %s/%s",
                        selected_date,
                        market
                    )
                    return 0

                if not rows:
                    logger.info(
                        "Captura de auditoria registrada sem elegíveis: %s/%s status=%s",
                        selected_date,
                        market,
                        availability_report_status
                    )
                    return 0

                cursor.executemany(
                    """
                    INSERT INTO audit_snapshots (
                        selected_date,
                        market,
                        player_id,
                        player_name,
                        team,
                        opponent,
                        position,
                        triage_score,
                        last5_avg,
                        last10_avg,
                        minutes_last5_avg,
                        audit_rank,
                        availability_status,
                        payload
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (
                        selected_date,
                        market,
                        player_id
                    ) DO NOTHING
                    """,
                    rows
                )

        logger.info(
            "Snapshot de auditoria processado: %s elegíveis enviados para %s/%s.",
            len(rows),
            selected_date,
            market
        )
        return len(rows)

    except Exception as error:
        logger.exception(
            "Falha ao salvar snapshot independente de auditoria: %s",
            error
        )
        return 0


@app.on_event("startup")
def startup_database():
    initialize_database()


HEADERS = {
    "User-Agent": "Mozilla/5.0 NBA Props Scanner"
}


SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/nba/scoreboard"
)


ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/nba/teams/{team}/roster"
)

TEAM_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/nba/teams/{team}/schedule"
)


GAMELOG_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/"
    "basketball/nba/athletes/{athlete}/gamelog"
)


SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/nba/summary"
)


MARKETS = {
    "points": {
        "index": 13,
        "label": "Pontos"
    },

    "assists": {
        "index": 8,
        "label": "Assistências"
    },

    "rebounds": {
        "index": 7,
        "label": "Rebotes"
    },

    "threes": {
        "index": 3,
        "label": "Tentativas de 3 pontos"
    },

    "two_pa": {
        "index": 14,
        "label": "Tentativas de 2 pontos"
    },
    "fgm": {
        "index": 2,
        "label": "Cestas convertidas"
    },
}

ACTIVE_MARKETS = {"points", "assists", "rebounds", "threes", "two_pa", "fgm"}


# ============================================================
# TEMPORADA
# ============================================================

def season_for(selected_date: date) -> str:

    start_year = (
        selected_date.year
        if selected_date.month >= 10
        else selected_date.year - 1
    )

    return (
        f"{start_year}-{str(start_year + 1)[-2:]}"
    )


def season_year(selected_date: date) -> int:

    return (
        selected_date.year + 1
        if selected_date.month >= 10
        else selected_date.year
    )


# ============================================================
# CONSULTA ESPN
# ============================================================

# Cada thread mantém sua própria sessão HTTP para reutilizar conexões
# sem compartilhar requests.Session entre threads.
_http_local = threading.local()


def _cache_metrics(cache):
    """Retorna métricas simples do cache para observabilidade de performance."""
    info = cache.cache_info()
    return {"hits": info.hits, "misses": info.misses, "currsize": info.currsize}


def get_http_session():
    session = getattr(_http_local, "session", None)

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _http_local.session = session

    return session


def request_json(
    url: str,
    params=None
) -> dict:

    try:

        response = get_http_session().get(
            url,
            params=params,
            timeout=(8, 25)
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        logger.warning(
            "Falha na consulta ESPN: %s",
            error
        )

        raise

    except ValueError as error:

        logger.warning(
            "JSON inválido recebido da ESPN: %s",
            error
        )

        raise


# ============================================================
# CONVERSÃO DE ESTATÍSTICAS
# ============================================================

def parse_number(value) -> float:

    try:
        return float(str(value).strip())

    except (ValueError, TypeError):
        return 0.0


def parse_minutes(value) -> float:

    if value is None:
        return 0.0

    value = str(value).strip()

    if ":" in value:

        try:

            minutes, seconds = value.split(":")

            return (
                float(minutes)
                + float(seconds) / 60
            )

        except (ValueError, TypeError):
            return 0.0

    return parse_number(value)


def parse_attempts(value) -> float:

    if value is None:
        return 0.0

    try:
        parts = str(value).strip().split("-")
        if len(parts) != 2:
            return 0.0
        return float(parts[1])
    except (ValueError, TypeError):
        return 0.0


def parse_made(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("displayValue", value.get("value"))
        if value is None:
            return None
    text = str(value).strip()
    if "-" not in text:
        return parse_optional_number(value)
    parts = text.split("-")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0])
    except (ValueError, TypeError):
        return None


def subtract_stats(total, threes):
    if total is None or threes is None:
        return None
    return max(0.0, float(total) - float(threes))


def parse_optional_number(value):

    if value is None:
        return None

    if isinstance(value, dict):

        value = (
            value.get("value")
            if value.get("value") is not None
            else value.get("displayValue")
        )

    if value in ("", "--", "-", "N/A"):
        return None

    try:
        return float(str(value).strip())

    except (ValueError, TypeError):
        return None


def parse_optional_attempts(value):

    if value is None:
        return None

    if isinstance(value, dict):

        display_value = value.get("displayValue")

        if display_value is not None:
            value = display_value

        else:
            value = value.get("value")

    if value in ("", "--", "-", "N/A"):
        return None

    text = str(value).strip()

    if "-" in text:

        parts = text.split("-")

        if len(parts) == 2:

            try:
                return float(parts[1])

            except (ValueError, TypeError):
                return None

    return parse_optional_number(value)


# ============================================================
# ESTATÍSTICAS DEFENSIVAS DE UMA PARTIDA
# ============================================================

def extract_team_defense_stats(
    summary,
    defending_team
):

    teams = (
        summary.get("boxscore", {})
        .get("teams", [])
    )

    opponent_stats = None

    defending_team = (
        defending_team or ""
    ).upper()

    for item in teams:

        team_abbr = (
            item.get("team", {})
            .get("abbreviation")
            or ""
        ).upper()

        if team_abbr != defending_team:

            opponent_stats = item.get(
                "statistics",
                []
            )

            break

    if opponent_stats is None:
        return None

    stats = {
        stat.get("name"): stat.get("displayValue")
        for stat in opponent_stats
    }

    fga_allowed = parse_attempts(stats.get("fieldGoalsMade-fieldGoalsAttempted"))    three_pa_allowed = parse_attempts(stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
    return {
        "fga_allowed": fga_allowed,
        "three_pa_allowed": three_pa_allowed,
        "two_pa_allowed": subtract_stats(fga_allowed, three_pa_allowed),
        "fgm_allowed": parse_made(stats.get("fieldGoalsMade-fieldGoalsAttempted")),
        "rebounds_allowed": parse_number(stats.get("totalRebounds")),
        "assists_allowed": parse_number(stats.get("assists"))
    }


def normalize_position(position_data):

    if isinstance(position_data, dict):

        position = (
            position_data.get("abbreviation")
            or position_data.get("displayName")
            or position_data.get("name")
            or ""
        )

    else:

        position = position_data or ""

    position = str(position).strip().upper()

    if position in ("G", "PG", "SG", "GUARD"):
        return "G"

    if position in ("F", "SF", "PF", "FORWARD"):
        return "F"

    if position in ("C", "CENTER"):
        return "C"

    return "OTHER"


def normalize_stat_key(key):

    if key is None:
        return ""

    key = str(key).strip()

    aliases = {
        "FG": "fieldGoalsMade-fieldGoalsAttempted",
        "3PT": "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
        "REB": "rebounds",
        "AST": "assists",
        "MIN": "minutes",
        "PTS": "points"
    }

    upper_key = key.upper()

    if upper_key in aliases:
        return aliases[upper_key]

    return key


def get_player_stat_value(
    raw_stats,
    key_indexes,
    stat_key,
    attempt_stat=False,
    made_stat=False
):

    normalized_key = normalize_stat_key(
        stat_key
    )

    value = None

    if isinstance(raw_stats, dict):

        value = raw_stats.get(
            normalized_key
        )

        if value is None:
            value = raw_stats.get(
                stat_key
            )

    elif isinstance(raw_stats, list):

        index = key_indexes.get(
            normalized_key
        )

        if index is None:
            return None

        if index >= len(raw_stats):
            return None

        value = raw_stats[index]

    if normalized_key == "minutes":

        if isinstance(value, dict):

            value = (
                value.get("displayValue")
                if value.get("displayValue") is not None
                else value.get("value")
            )

        if value in ("", "--", "-", "N/A", None):
            return None

        return parse_minutes(value)

    if attempt_stat:
        return parse_optional_attempts(value)

    if made_stat:
        return parse_made(value)

    return parse_optional_number(value)


def get_athlete_position(athlete):

    if not isinstance(athlete, dict):
        return "OTHER"

    position_data = athlete.get(
        "position",
        {}
    )

    if not position_data:
        position_data = athlete.get(
            "positionName",
            ""
        )

    return normalize_position(
        position_data
    )


def extract_opponent_players_by_position(
    summary,
    defending_team
):

    result = {
        "G": [],
        "F": [],
        "C": [],
        "OTHER": []
    }

    boxscore = summary.get("boxscore", {})
    teams = boxscore.get("teams", [])
    players_blocks = boxscore.get("players", [])

    defending_team = (defending_team or "").upper()
    opponent_team = None

    for team in teams:

        team_abbr = (
            team.get("team", {})
            .get("abbreviation")
            or ""
        ).upper()

        if team_abbr and team_abbr != defending_team:
            opponent_team = team_abbr
            break

    if not opponent_team:
        return {
            "opponent": None,
            "players": result
        }

    opponent_players_block = None

    for block in players_blocks:

        team_abbr = (
            block.get("team", {})
            .get("abbreviation")
            or ""
        ).upper()

        if team_abbr == opponent_team:
            opponent_players_block = block
            break

    if not opponent_players_block:
        return {
            "opponent": opponent_team,
            "players": result
        }

    statistics_blocks = (
        opponent_players_block.get(
            "statistics",
            []
        )
    )

    if isinstance(statistics_blocks, dict):
        statistics_blocks = [statistics_blocks]

    if not isinstance(statistics_blocks, list):
        return {
            "opponent": opponent_team,
            "players": result
        }

    for statistics_block in statistics_blocks:

        if not isinstance(statistics_block, dict):
            continue

        keys = (
            statistics_block.get("keys")
            or statistics_block.get("names")
            or statistics_block.get("labels")
            or []
        )

        key_indexes = {}

        for index, key in enumerate(keys):

            normalized_key = normalize_stat_key(
                key
            )

            if normalized_key:
                key_indexes[normalized_key] = index

        athletes = statistics_block.get(
            "athletes",
            []
        )

        if not isinstance(athletes, list):
            continue

        for athlete_entry in athletes:

            if not isinstance(athlete_entry, dict):
                continue

            athlete = athlete_entry.get(
                "athlete",
                {}
            )

            if not isinstance(athlete, dict):
                continue

            player_name = (
                athlete.get("displayName")
                or athlete.get("fullName")
                or athlete.get("shortName")
            )

            if not player_name:
                continue

            raw_stats = (
                athlete_entry.get("stats")
                or athlete_entry.get("statistics")
            )

            if not raw_stats:
                continue

            position_group = get_athlete_position(
                athlete
            )

            points = get_player_stat_value(
                raw_stats,
                key_indexes,
                "points"
            )

            assists = get_player_stat_value(
                raw_stats,
                key_indexes,
                "assists"
            )

            rebounds = get_player_stat_value(
                raw_stats,
                key_indexes,
                "rebounds"
            )

            fga = get_player_stat_value(
                raw_stats,
                key_indexes,
                "fieldGoalsMade-fieldGoalsAttempted",
                attempt_stat=True
            )

            three_pa = get_player_stat_value(
                raw_stats,
                key_indexes,
                "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                attempt_stat=True
            )
            fgm = get_player_stat_value(
                raw_stats,
                key_indexes,
                "fieldGoalsMade-fieldGoalsAttempted",
                made_stat=True
            )
            two_pa = subtract_stats(fga, three_pa)

            minutes = get_player_stat_value(
                raw_stats,
                key_indexes,
                "minutes"
            )

            if minutes is not None and minutes <= 0:
                continue

            if (
                minutes is None
                and all(
                    v is None
                    for v in (
                        points,
                        assists,
                        rebounds,
                        fga,
                        three_pa,
                        two_pa,
                        fgm
                    )
                )
            ):
                continue

            result[position_group].append({
                "player": player_name,
                "position": position_group,
                "minutes": minutes,
                "points": points,
                "assists": assists,
                "rebounds": rebounds,
                "fga": fga,
                "three_pa": three_pa,
                "two_pa": two_pa,
                "fgm": fgm
            })

    return {
        "opponent": opponent_team,
        "players": result
    }


def aggregate_position_game(
    summary,
    defending_team
):

    extracted = extract_opponent_players_by_position(
        summary,
        defending_team
    )

    players_by_position = extracted["players"]
    result = {}

    stat_keys = [
        "points",
        "assists",
        "rebounds",
        "fga",
        "three_pa",
        "two_pa",
        "fgm"
    ]

    for position in ("G", "F", "C", "OTHER"):

        players = players_by_position.get(
            position,
            []
        )

        position_result = {
            "players_count": len(players),
            "points": None,
            "assists": None,
            "rebounds": None,
            "fga": None,
            "three_pa": None,
            "two_pa": None,
            "fgm": None,
            "stat_players_count": {},
            "players": players
        }

        for stat_key in stat_keys:

            values = [
                p[stat_key]
                for p in players
                if p.get(stat_key) is not None
            ]

            position_result["stat_players_count"][
                stat_key
            ] = len(values)

            if values:
                position_result[stat_key] = round(
                    sum(values),
                    1
                )

        result[position] = position_result

    return {
        "opponent": extracted["opponent"],
        "positions": result
    }


def calculate_minutes_filtered_position_defense(
    sample,
    position,
    stat_key,
    minutes_threshold
):

    total_stat = 0.0
    total_minutes = 0.0
    player_appearances = 0
    games_analyzed = 0

    for game in sample:

        position_data = (
            game.get("positions", {})
            .get(position, {})
        )

        players = position_data.get(
            "players",
            []
        )

        game_has_valid_player = False

        for player in players:

            player_minutes = player.get(
                "minutes"
            )

            stat_value = player.get(
                stat_key
            )

            if (
                player_minutes is None
                or player_minutes < minutes_threshold
                or stat_value is None
            ):
                continue

            total_stat += stat_value
            total_minutes += player_minutes
            player_appearances += 1
            game_has_valid_player = True

        if game_has_valid_player:
            games_analyzed += 1

    average_per_player = (
        round(
            total_stat / player_appearances,
            2
        )
        if player_appearances > 0
        else None
    )
    per_minute = (
        round(
            total_stat / total_minutes,            4
        )
        if total_minutes > 0
        else None
    )

    projected_value = (
        round(
            per_minute * minutes_threshold,
            2
        )
        if per_minute is not None
        else None
    )

    return {
        "minutes_threshold": round(
            minutes_threshold,
            1
        ),
        "games_analyzed": games_analyzed,
        "player_appearances": player_appearances,
        "total_minutes": round(
            total_minutes,
            1
        ),
        "total_stat": round(
            total_stat,
            1
        ),
        "average_per_player_appearance": (
            average_per_player
        ),
        "per_minute": per_minute,
        "projected_at_player_minutes": (
            projected_value
        )
    }


def calculate_minutes_band_position_defense(
    sample,
    position,
    stat_key,
    minutes_center,
    margin=5.0
):
    """
    Média individual por aparição para atletas da mesma posição
    dentro de uma faixa simétrica de minutagem (centro ± margem).
    Aceita amostras defensivas de uma equipe e amostras agregadas da liga.
    """
    lower_minutes = max(0.0, float(minutes_center) - float(margin))
    upper_minutes = float(minutes_center) + float(margin)

    total_stat = 0.0
    total_minutes = 0.0
    player_appearances = 0
    games_analyzed = 0

    for game in sample:
        position_data = (
            game.get("positions", {})
            .get(position, {})
        )
        players = position_data.get("players", [])
        game_has_valid_player = False

        for player in players:
            player_minutes = player.get("minutes")
            stat_value = player.get(stat_key)

            if (
                player_minutes is None
                or player_minutes < lower_minutes
                or player_minutes > upper_minutes
                or stat_value is None
            ):
                continue

            total_stat += stat_value
            total_minutes += player_minutes
            player_appearances += 1
            game_has_valid_player = True

        if game_has_valid_player:
            games_analyzed += 1

    return {
        "minutes_center": round(float(minutes_center), 1),
        "minutes_margin": round(float(margin), 1),
        "minutes_lower": round(lower_minutes, 1),
        "minutes_upper": round(upper_minutes, 1),
        "games_analyzed": games_analyzed,
        "player_appearances": player_appearances,
        "total_minutes": round(total_minutes, 1),
        "total_stat": round(total_stat, 1),
        "average_per_player_appearance": (
            round(total_stat / player_appearances, 2)
            if player_appearances else None
        )
    }


def calculate_equal_game_position_defense(sample, position, stat_key):
    """Média da produção por aparição dentro de cada jogo; jogos têm peso igual."""
    game_averages = []
    appearances = 0
    for game in sample:
        position_data = game.get("positions", {}).get(position, {})
        players = position_data.get("players", [])
        values = [p.get(stat_key) for p in players if p.get(stat_key) is not None]
        if values:
            game_averages.append(sum(values) / len(values))
            appearances += len(values)
    return {
        "average_per_game": round(sum(game_averages) / len(game_averages), 2) if game_averages else None,
        "games_analyzed": len(game_averages),
        "player_appearances": appearances
    }

def get_completed_team_games(
    schedule_data,
    team_abbr,
    limit=10
):

    games = []

    for event in schedule_data.get(
        "events",
        []
    ):

        season_type = (
            event.get("seasonType", {})
            .get("abbreviation")
        )

        if season_type != "reg":
            continue

        competition = (
            event.get("competitions", [{}])[0]
        )

        status = (
            competition.get("status", {})
            .get("type", {})
        )

        if not status.get("completed"):
            continue

        opponent = None
        home_away = None

        for competitor in competition.get(
            "competitors",
            []
        ):

            competitor_team = (
                competitor.get("team", {})
                .get("abbreviation")
            )

            if competitor_team == team_abbr:
                home_away = competitor.get("homeAway")
            else:
                opponent = competitor_team

        if not opponent or home_away not in {"home", "away"}:
            continue

        games.append({
            "game_id": str(event.get("id")),
            "date": event.get("date"),
            "opponent": opponent,
            "home_away": home_away
        })

    games.sort(
        key=lambda game: game.get("date") or "",
        reverse=True
    )

    return games[:limit]


@lru_cache(maxsize=500)
def _get_cached_game_summary(
    event_id: str
) -> dict:

    return request_json(
        SUMMARY_URL,
        {"event": event_id}
    )


# Evita que equipes diferentes disparem simultaneamente a mesma consulta
# antes que o resultado seja armazenado no cache. Locks por faixa mantêm
# o controle de memória limitado, sem serializar todas as partidas.
_GAME_SUMMARY_LOCKS = tuple(
    threading.Lock()
    for _ in range(64)
)

# Prewarm opcional do histórico de atletas. É disparado pela tela de jogos
# antes da seleção do mercado, para que a primeira pesquisa possa aproveitar
# o cache sem tornar o endpoint /games lento.
_PLAYER_HISTORY_PREWARM_LOCK = threading.Lock()
_PLAYER_HISTORY_PREWARM_STATE = {}


def _player_history_prewarm_worker(selected_date, team_abbreviations, year, event):
    try:
        started = time.perf_counter()
        players_by_id = {}

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(get_team_roster, team): team
                for team in team_abbreviations
            }
            for future in as_completed(futures):
                try:
                    roster = future.result()
                    for player in roster:
                        players_by_id[str(player["id"])] = player
                except Exception as error:
                    logger.warning(
                        "Prewarm roster falhou em %s/%s: %s",
                        selected_date,
                        futures[future],
                        error
                    )

        players = list(players_by_id.values())
        successful = 0
        errors = 0

        def fetch_player(player):
            try:
                get_athlete_logs(player["id"], year)
                return True
            except Exception as error:
                logger.warning(
                    "Prewarm histórico falhou para %s/%s: %s",
                    selected_date,
                    player.get("player"),
                    error
                )
                return False

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(fetch_player, player)
                for player in players
            ]
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    errors += 1

        elapsed = time.perf_counter() - started
        with _PLAYER_HISTORY_PREWARM_LOCK:
            state = _PLAYER_HISTORY_PREWARM_STATE.get(selected_date)
            if state:
                state["status"] = "ok"
                state["elapsed"] = elapsed
                state["players"] = len(players)
                state["successful"] = successful
                state["errors"] = errors
        logger.info(
            "Player history prewarm: data=%s status=ok jogadores=%s sucesso=%s erros=%s tempo=%.2fs",
            selected_date,
            len(players),
            successful,
            errors,
            elapsed
        )
    except Exception as error:
        with _PLAYER_HISTORY_PREWARM_LOCK:
            state = _PLAYER_HISTORY_PREWARM_STATE.get(selected_date)
            if state:
                state["status"] = "error"
                state["elapsed"] = time.perf_counter() - state["started"]
                state["error"] = str(error)
        logger.warning(
            "Player history prewarm falhou para %s: %s",
            selected_date,
            error
        )
    finally:
        event.set()


def schedule_player_history_prewarm(selected_date, team_abbreviations):
    year = season_year(
        datetime.strptime(selected_date, "%Y-%m-%d").date()
    )
    with _PLAYER_HISTORY_PREWARM_LOCK:
        current = _PLAYER_HISTORY_PREWARM_STATE.get(selected_date)
        if current and current.get("status") in {"running", "ok"}:
            return current

        event = threading.Event()
        state = {
            "status": "running",
            "started": time.perf_counter(),
            "event": event,
        }
        _PLAYER_HISTORY_PREWARM_STATE[selected_date] = state

    thread = threading.Thread(
        target=_player_history_prewarm_worker,
        args=(selected_date, tuple(sorted(team_abbreviations)), year, event),
        daemon=True,
        name=f"player-history-prewarm-{selected_date}"
    )
    thread.start()
    return state


def wait_player_history_prewarm(selected_date):
    with _PLAYER_HISTORY_PREWARM_LOCK:
        state = _PLAYER_HISTORY_PREWARM_STATE.get(selected_date)
        event = state.get("event") if state else None

    if event is None:
        return

    wait_started = time.perf_counter()
    completed = event.wait(timeout=45)
    waited = time.perf_counter() - wait_started
    if waited > 0.01:
        logger.info(
            "Player history prewarm wait: data=%s wait=%.2fs status=%s completed=%s",
            selected_date,
            waited,
            state.get("status", "unknown") if state else "unknown",
            completed
        )
    if not completed:
        logger.warning(
            "Player history prewarm excedeu 45s; Scanner seguirá sem espera ilimitada: data=%s",
            selected_date
        )




def get_game_summary(
    event_id: str
) -> dict:

    event_id = str(event_id)
    lock = _GAME_SUMMARY_LOCKS[
        hash(event_id) % len(_GAME_SUMMARY_LOCKS)
    ]

    with lock:
        return _get_cached_game_summary(event_id)


@lru_cache(maxsize=60)
def get_team_defense_history(
    team_abbr,
    limit=10,
    target_date=None,
    force_refresh=False
):

    if target_date is None:
        target_date = date.today()

    if force_refresh:
        schedule = request_json(
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"basketball/nba/teams/{team_abbr.lower()}/schedule",
            {"season": season_year(target_date)}
        )
        schedule_games = get_completed_team_games(
            schedule,
            team_abbr.upper(),
            1000
        )
    else:
        schedule_games = get_team_regular_season_schedule(
            team_abbr.upper(),
            season_year(target_date)
        )

    games = [
        game
        for game in schedule_games
        if (
            game.get("date") or ""
        )[:10] < target_date.isoformat()
    ]

    games.sort(
        key=lambda game: game.get("date") or "",
        reverse=True
    )

    games = games[:limit]
    results = []

    def fetch_game_defense(game):

        # Mesmo em refresh, o boxscore concluído é imutável para a data.
        # Usa o cache compartilhado para que o contexto pré-aqueça os
        # mesmos summaries aproveitados pela primeira consulta do Scanner.
        summary = get_game_summary(
            str(game["game_id"])
        )

        stats = extract_team_defense_stats(
            summary,
            team_abbr
        )

        if not stats:
            return None

        position_data = aggregate_position_game(
            summary,
            team_abbr
        )

        return {
            **game,
            **stats,
            "positions": position_data["positions"]
        }

    # Cada histórico de equipe consulta até 10 boxscores. Mantemos
    # o mesmo teto de 12 consultas simultâneas à ESPN, mas distribuímos
    # melhor a carga entre equipes para reduzir a latência do primeiro scan.
    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        futures = {
            executor.submit(
                fetch_game_defense,
                game
            ): game
            for game in games
        }

        for future in as_completed(futures):

            game = futures[future]

            try:

                result = future.result()

                if result:
                    results.append(result)

            except Exception as error:

                logger.warning(
                    "Falha no histórico defensivo %s - jogo %s: %s",
                    team_abbr,
                    game.get("game_id"),
                    error
                )

    results.sort(
        key=lambda game: game.get("date") or "",
        reverse=True
    )

    stat_keys = [
        "fga_allowed",
        "three_pa_allowed",
        "two_pa_allowed",
        "fgm_allowed",
        "rebounds_allowed",
        "assists_allowed"
    ]

    position_stat_keys = [
        "points",
        "assists",
        "rebounds",
        "fga",
        "three_pa",
        "two_pa",
        "fgm"    ]

    def calculate_averages(sample):
        averages = {}

        for key in stat_keys:

            values = [
                game[key]
                for game in sample
                if game.get(key) is not None
            ]

            averages[key] = (
                round(
                    sum(values) / len(values),
                    1
                )
                if values
                else None
            )

        return averages

    def calculate_position_averages(sample):

        position_results = {}

        for position in ("G", "F", "C", "OTHER"):

            position_games = [
                game
                for game in sample
                if (
                    game.get("positions", {})
                    .get(position, {})
                    .get("players_count", 0) > 0
                )
            ]

            averages = {}
            individual_averages = {}
            stat_games_analyzed = {}
            stat_player_appearances = {}

            for stat_key in position_stat_keys:

                game_values = []
                individual_total = 0.0
                individual_appearances = 0

                for game in position_games:

                    position_data = (
                        game.get("positions", {})
                        .get(position, {})
                    )

                    value = position_data.get(
                        stat_key
                    )

                    if value is not None:
                        game_values.append(value)
                        individual_total += value

                    player_count = (
                        position_data
                        .get("stat_players_count", {})
                        .get(stat_key, 0)
                    )

                    individual_appearances += player_count

                stat_games_analyzed[
                    stat_key
                ] = len(game_values)

                averages[stat_key] = (
                    round(
                        sum(game_values)
                        / len(game_values),
                        1
                    )
                    if game_values
                    else None
                )

                stat_player_appearances[
                    stat_key
                ] = individual_appearances

                individual_averages[
                    stat_key
                ] = (
                    round(
                        individual_total
                        / individual_appearances,
                        2
                    )
                    if individual_appearances > 0
                    else None
                )

            players_per_game = [
                game.get("positions", {})
                .get(position, {})
                .get("players_count", 0)
                for game in position_games
            ]

            position_results[position] = {
                "games_analyzed": len(
                    position_games
                ),
                "players_per_game_avg": (
                    round(
                        sum(players_per_game)
                        / len(players_per_game),
                        1
                    )
                    if players_per_game
                    else 0
                ),
                "stat_games_analyzed": (
                    stat_games_analyzed
                ),
                "averages": averages,
                "individual_averages": (
                    individual_averages
                ),
                "stat_player_appearances": (
                    stat_player_appearances
                )
            }

        return position_results

    history = {
        "team": team_abbr,
        "games_analyzed": len(results),
        "games": results,
        "averages": calculate_averages(results),
        "positions": calculate_position_averages(results)
    }

    for window in (5, 10):

        sample = results[:window]

        history[f"L{window}"] = {
            "team": team_abbr,
            "games_analyzed": len(sample),
            "games": sample,
            "averages": calculate_averages(sample),
            "positions": calculate_position_averages(sample)
        }

    return history


def get_team_defense_history_fresh(
    team_abbr,
    limit=10,
    target_date=None
):
    """Atualiza a Central sem ler/escrever nos caches compartilhados do Scanner."""
    return get_team_defense_history.__wrapped__(
        team_abbr,
        limit,
        target_date,
        True
    )


@lru_cache(maxsize=120)
def fetch_team_schedule_cached(
    team_abbr: str,
    season: int,
    cache_window: int
) -> tuple:
    """Consulta a agenda da equipe e mantém cache curto para reduzir chamadas à ESPN."""
    data = request_json(
        TEAM_SCHEDULE_URL.format(team=team_abbr.lower()),
        {"season": season}
    )
    completed_games = []
    for event in data.get("events", []):
        event_date = (event.get("date") or "")[:10]
        competitions = event.get("competitions") or []
        if not event_date or not competitions:
            continue
        competition = competitions[0]
        status_type = (
            competition.get("status", {}).get("type", {})
            or event.get("status", {}).get("type", {})
        )
        if not status_type.get("completed", False):
            continue
        competitors = competition.get("competitors") or []
        own = next((
            item for item in competitors
            if (item.get("team") or {}).get("abbreviation", "").upper()
            == team_abbr.upper()
        ), None)
        if own is None:
            continue
        opponent = next((item for item in competitors if item is not own), None)
        if opponent is None:
            continue
        if own.get("winner") is True:
            result = "W"
        elif opponent.get("winner") is True:
            result = "L"
        else:
            try:
                own_score = float(own.get("score"))
                opponent_score = float(opponent.get("score"))
                if own_score == opponent_score:
                    continue
                result = "W" if own_score > opponent_score else "L"
            except (TypeError, ValueError):
                continue
        completed_games.append((event_date, result))
    completed_games.sort(key=lambda item: item[0])
    return tuple(completed_games)


def fetch_team_recent_form(team_abbr: str, selected_date: str) -> dict:
    """Retorna os cinco resultados concluídos anteriores à data pesquisada."""
    started = time.perf_counter()
    try:
        target_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
        cache_window = int(time.time() // 900)
        schedule = fetch_team_schedule_cached(
            team_abbr, season_year(target_date), cache_window
        )
        schedule_elapsed = time.perf_counter() - started
        logger.info(
            "Games team schedule: data=%s equipe=%s tempo=%.2fs jogos=%s",
            selected_date, team_abbr, schedule_elapsed, len(schedule)
        )
        previous_games = [
            result for game_date, result in schedule
            if game_date < selected_date
        ][-5:]
        wins = previous_games.count("W")
        losses = previous_games.count("L")
        return {
            "last5": previous_games,
            "last5_record": f"{wins}-{losses}",
            "last5_games": len(previous_games),
            "last5_status": "ok" if previous_games else "unavailable"
        }
    except Exception as error:
        logger.warning(
            "Não foi possível consultar retrospecto recente de %s: %s",
            team_abbr, error
        )
        return {
            "last5": [], "last5_record": None,
            "last5_games": 0, "last5_status": "unavailable"
        }


def fetch_espn_games(
    selected_date: str,
    enrich_context: bool = True,
    prewarm_player_history: bool = False
) -> dict:

    try:

        data = request_json(
            SCOREBOARD_URL,
            {
                "dates": selected_date.replace(
                    "-",
                    ""
                )
            }
        )

    except Exception:

        raise HTTPException(
            status_code=503,
            detail=(
                "Não foi possível consultar "
                "a agenda da ESPN."
            )
        )

    games = []

    for event in data.get("events", []):

        competitions = event.get(
            "competitions",
            []
        )

        if not competitions:
            continue

        home = None
        away = None

        for competitor in competitions[0].get(
            "competitors",
            []
        ):

            team = competitor.get(
                "team",
                {}
            )

            team_logos = team.get("logos") or []
            logo_url = (
                team_logos[0].get("href")
                if team_logos and isinstance(team_logos[0], dict)
                else team.get("logo")
            )
            item = {
                "abbr": team.get(
                    "abbreviation",
                    ""
                ),
                "name": team.get(
                    "displayName",
                    ""
                ),
                "logo": logo_url
            }

            if competitor.get("homeAway") == "home":
                home = item
            elif competitor.get("homeAway") == "away":
                away = item

        if home and away:

            # A ESPN fornece o início em UTC; a interface exibe horário de Brasília.
            event_date = event.get("date")
            start_time_brt = None
            start_date_et = None
            start_time_et = None

            if event_date:
                try:
                    start_utc = datetime.fromisoformat(
                        event_date.replace("Z", "+00:00")
                    )
                    start_time_brt = start_utc.astimezone(
                        ZoneInfo("America/Sao_Paulo")
                    ).strftime("%H:%M")
                    start_et = start_utc.astimezone(
                        ZoneInfo("America/New_York")
                    )
                    start_date_et = start_et.strftime("%Y-%m-%d")
                    start_time_et = start_et.strftime("%H:%M")
                except (ValueError, TypeError):
                    logger.warning(
                        "Horário inválido recebido da ESPN para jogo %s: %s",
                        event.get("id"),
                        event_date
                    )

            games.append({
                "game_id": event.get("id"),
                "home": home,
                "away": away,
                "start_time_brt": start_time_brt,
                "start_date_et": start_date_et,
                "start_time_et": start_time_et,
                "status": (
                    event.get("status", {})
                    .get("type", {})
                    .get(
                        "description",
                        "Não informado"
                    )
                ),
                "completed": bool(
                    event.get("status", {})
                    .get("type", {})
                    .get(
                        "completed",
                        False
                    )
                )
            })

    # A agenda completa usa contexto recente e prewarm defensivo para a interface.
    # A coleta de resultados precisa apenas do placar/confrontos e deve evitar
    # esse trabalho extra, pois o endpoint é chamado para cada data/mercado.
    if enrich_context:
        team_abbreviations = {
            team.get("abbr")
            for game in games
            for team in (game.get("home"), game.get("away"))
            if team and team.get("abbr")
        }
        team_forms = {}
        games_context_start = time.perf_counter()
        recent_form_cache_before = _cache_metrics(fetch_team_schedule_cached)
        if team_abbreviations:
            with ThreadPoolExecutor(max_workers=min(10, len(team_abbreviations))) as executor:
                futures = {
                    executor.submit(fetch_team_recent_form, team_abbr, selected_date): team_abbr
                    for team_abbr in team_abbreviations
                }
                for future in as_completed(futures):
                    team_abbr = futures[future]
                    try:
                        team_forms[team_abbr] = future.result()
                    except Exception as error:
                        logger.warning("Falha inesperada no retrospecto de %s: %s", team_abbr, error)
                        team_forms[team_abbr] = {
                            "last5": [], "last5_record": None,
                            "last5_games": 0, "last5_status": "unavailable"
                        }

        games_context_elapsed = time.perf_counter() - games_context_start
        recent_form_cache_after = _cache_metrics(fetch_team_schedule_cached)
        logger.info(
            "Games timing: data=%s equipes=%s tempo_contexto=%.2fs cache_hits_delta=%s cache_misses_delta=%s cache_size=%s",
            selected_date,
            len(team_abbreviations),
            games_context_elapsed,
            recent_form_cache_after["hits"] - recent_form_cache_before["hits"],
            recent_form_cache_after["misses"] - recent_form_cache_before["misses"],
            recent_form_cache_after["currsize"]
        )

        for game in games:
            for side in ("home", "away"):
                team = game.get(side)
                if team:
                    team.update(team_forms.get(team.get("abbr"), {
                        "last5": [], "last5_record": None,
                        "last5_games": 0, "last5_status": "unavailable"
                    }))

        # O benchmark defensivo é carregado sob demanda pelo fluxo de
        # candidatos. Não iniciar um prewarm paralelo durante a consulta da agenda.

    return {
        "date": selected_date,
        "count": len(games),
        "source": "ESPN",
        "games": games
    }


def collect_roster_players(
    node,
    players
):

    if isinstance(node, dict):

        athlete_id = node.get("id")
        name = node.get("fullName")

        if athlete_id and name:

            athlete_id = str(athlete_id)
            position = get_athlete_position(
                node
            )

            existing = players.get(
                athlete_id
            )

            if existing is None:
                players[athlete_id] = {
                    "id": athlete_id,
                    "player": name,
                    "position": position
                }
            elif (
                existing.get("position") == "OTHER"
                and position != "OTHER"
            ):
                existing["position"] = position

        for value in node.values():
            collect_roster_players(
                value,
                players
            )

    elif isinstance(node, list):

        for value in node:
            collect_roster_players(                value,
                players
            )

@lru_cache(maxsize=40)
def get_team_roster(
    team: str
) -> tuple:

    data = request_json(
        ROSTER_URL.format(
            team=team.lower()
        )
    )

    players = {}
    collect_roster_players(data, players)

    return tuple(players.values())


def get_players_for_date(
    selected_date: str
) -> dict:

    # A coleta de candidatos precisa apenas da agenda para identificar
    # equipes e elencos. O retrospecto recente é exibido na tela de jogos
    # e não deve ser consultado novamente dentro da pesquisa.
    schedule = fetch_espn_games(
        selected_date,
        enrich_context=False
    )

    teams = set()

    for game in schedule["games"]:
        teams.add(game["home"]["abbr"])
        teams.add(game["away"]["abbr"])

    memberships = {}

    with ThreadPoolExecutor(
        max_workers=6
    ) as executor:

        futures = {
            executor.submit(
                get_team_roster,
                team
            ): team
            for team in teams
        }

        for future in as_completed(futures):

            team = futures[future]

            try:

                roster = future.result()

                for player in roster:

                    athlete_id = str(
                        player["id"]
                    ).strip()

                    entry = memberships.setdefault(
                        athlete_id,
                        {
                            "player": player,
                            "teams": set()
                        }
                    )

                    entry["teams"].add(team)

                    if (
                        entry["player"].get(
                            "position"
                        ) == "OTHER"
                        and player.get(
                            "position"
                        ) != "OTHER"
                    ):
                        entry["player"][
                            "position"
                        ] = player["position"]

            except Exception as error:

                logger.warning(
                    "Falha no elenco %s: %s",
                    team,
                    error
                )

    players = []
    duplicates = 0

    for athlete_id, entry in memberships.items():

        if len(entry["teams"]) != 1:

            duplicates += 1
            continue

        team = next(
            iter(entry["teams"])
        )

        players.append({
            **entry["player"],
            "team": team
        })

    players.sort(
        key=lambda p: p["player"].casefold()
    )

    return {
        "schedule": schedule,
        "players": players,
        "teams_found": len(teams),
        "players_excluded_multiple_teams": duplicates
    }


@lru_cache(maxsize=800)
def get_athlete_logs(
    athlete_id: str,
    year: int
) -> tuple:

    gamelog_request_started = time.perf_counter()
    try:
        data = request_json(
            GAMELOG_URL.format(
                athlete=athlete_id
            ),
            {"season": year}
        )
    finally:
        _http_local.last_gamelog_http_elapsed = (
            time.perf_counter() - gamelog_request_started
        )

    season_label = (
        f"{year - 1}-{str(year)[-2:]} Regular Season"
    )

    season_category = None

    for season_type in data.get(
        "seasonTypes",
        []
    ):

        display_name = season_type.get(
            "displayName",
            ""
        )

        if (
            display_name.strip().lower()
            == season_label.lower()
        ):
            season_category = season_type
            break

    if not season_category:
        return tuple()

    event_dates = data.get("events", {})
    games = {}

    for month in season_category.get(
        "categories",
        []
    ):

        for event in month.get("events", []):

            event_id = str(
                event.get("eventId", "")
            )

            stats = event.get(
                "stats",
                []
            )

            if not event_id or len(stats) <= 13:
                continue

            metadata = event_dates.get(
                event_id,
                {}
            )

            game_date = (
                metadata.get("gameDate")
                or metadata.get("date")
            )

            if not game_date:
                continue

            game_date = str(
                game_date
            )[:10]

            if len(game_date) != 10:
                continue

            games[event_id] = {
                "event_id": event_id,
                "date": game_date,
                "team": (
                    metadata.get("team") or {}
                ).get("abbreviation"),
                "opponent": (
                    metadata.get("opponent") or {}
                ).get("abbreviation"),
                "home_away": (
                    str(metadata.get("homeAway") or "").strip().lower()
                    if str(metadata.get("homeAway") or "").strip().lower() in {"home", "away"}
                    else None
                ),
                "minutes": parse_minutes(
                    stats[0]
                ),
                "fga": parse_attempts(stats[1]),
                "three_pa": parse_attempts(stats[3]),
                "two_pa": max(0.0, parse_attempts(stats[1]) - parse_attempts(stats[3])),
                "fgm": parse_made(stats[1]),
                "rebounds": parse_number(
                    stats[7]
                ),
                "assists": parse_number(
                    stats[8]
                ),
                "points": parse_number(
                    stats[13]
                )
            }

    # O endpoint de game log nem sempre inclui homeAway em events.
    # Completa o mando pela agenda oficial da equipe, que já expõe
    # competitor.homeAway e é compartilhada em cache por equipe/temporada.
    teams_missing_venue = {
        game.get("team")
        for game in games.values()
        if not game.get("home_away") and game.get("team")
    }

    for team_abbr in teams_missing_venue:
        try:
            team_schedule = get_team_regular_season_schedule(
                team_abbr.upper(),
                year
            )
            venue_by_event = {
                str(team_game.get("game_id")): team_game.get("home_away")
                for team_game in team_schedule
                if team_game.get("home_away") in {"home", "away"}
            }

            for game in games.values():
                if (
                    game.get("team", "").upper() == team_abbr.upper()
                    and not game.get("home_away")
                ):
                    game["home_away"] = venue_by_event.get(
                        str(game.get("event_id"))
                    )
        except Exception as error:
            logger.warning(
                "Não foi possível completar mando do game log para %s/%s: %s",
                team_abbr,
                year,
                error
            )

    ordered = sorted(
        games.values(),
        key=lambda game: game["date"],
        reverse=True
    )

    return tuple(ordered)


def normalize_diagnostic_player_name(name):
    """
    Normalização leve usada somente no diagnóstico de auditoria.
    Não participa da elegibilidade nem da persistência de resultados.
    """
    if name is None:
        return ""
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def find_player_in_summary_for_diagnostic(
    summary,
    snapshot_player_id,
    snapshot_player_name
):
    """
    Procura o atleta no boxscore primeiro pelo ID e, somente para diagnóstico,
    por nome normalizado. Não altera a regra oficial de coleta.
    """
    players_blocks = (
        summary.get("boxscore", {})
        .get("players", [])
    )
    if not isinstance(players_blocks, list):
        return None

    snapshot_id = str(snapshot_player_id or "").strip()
    snapshot_name = normalize_diagnostic_player_name(snapshot_player_name)
    name_matches = []

    for team_block in players_blocks:
        if not isinstance(team_block, dict):
            continue
        team_abbr = (
            team_block.get("team", {}).get("abbreviation")
            or ""
        ).upper()
        statistics_blocks = team_block.get("statistics", [])
        if isinstance(statistics_blocks, dict):
            statistics_blocks = [statistics_blocks]
        if not isinstance(statistics_blocks, list):
            continue

        for statistics_block in statistics_blocks:
            if not isinstance(statistics_block, dict):
                continue
            keys = (
                statistics_block.get("keys")
                or statistics_block.get("names")
                or statistics_block.get("labels")
                or []
            )
            key_indexes = {}
            for index, key in enumerate(keys):
                normalized_key = normalize_stat_key(key)
                if normalized_key:
                    key_indexes[normalized_key] = index

            athletes = statistics_block.get("athletes", [])
            if not isinstance(athletes, list):
                continue

            for athlete_entry in athletes:
                if not isinstance(athlete_entry, dict):
                    continue
                athlete = athlete_entry.get("athlete", {})
                if not isinstance(athlete, dict):
                    continue

                boxscore_id = str(athlete.get("id", "")).strip()
                boxscore_name = (
                    athlete.get("displayName")
                    or athlete.get("fullName")
                    or athlete.get("shortName")
                    or ""
                )
                raw_stats = (
                    athlete_entry.get("stats")
                    or athlete_entry.get("statistics")
                )

                minutes = None
                if raw_stats is not None:
                    minutes = get_player_stat_value(
                        raw_stats,
                        key_indexes,
                        "minutes"
                    )

                min_index = key_indexes.get("minutes")
                raw_min_value = None
                if isinstance(raw_stats, list) and min_index is not None and min_index < len(raw_stats):
                    raw_min_value = raw_stats[min_index]
                elif isinstance(raw_stats, dict):
                    raw_min_value = raw_stats.get("minutes", raw_stats.get("MIN"))

                candidate = {
                    "boxscore_player_id": boxscore_id,
                    "boxscore_player_name": boxscore_name,
                    "boxscore_team": team_abbr,
                    "boxscore_minutes": minutes,
                    "boxscore_statistics_keys": keys,
                    "boxscore_stats_type": type(raw_stats).__name__ if raw_stats is not None else "ausente",
                    "boxscore_stats_length": len(raw_stats) if isinstance(raw_stats, (list, dict)) else None,
                    "boxscore_min_index": min_index,
                    "boxscore_raw_min_value": raw_min_value,
                    "boxscore_athlete_entry_keys": list(athlete_entry.keys()),
                    "boxscore_athlete_keys": list(athlete.keys()),
                    "boxscore_did_not_play": athlete_entry.get("didNotPlay") or athlete_entry.get("did_not_play") or athlete_entry.get("reason")
                }

                if boxscore_id and boxscore_id == snapshot_id:
                    candidate["match_method"] = "id"
                    return candidate

                if (
                    snapshot_name
                    and normalize_diagnostic_player_name(boxscore_name)
                    == snapshot_name
                ):
                    name_matches.append(candidate)

    if name_matches:
        name_matches[0]["match_method"] = "name"
        return name_matches[0]

    return None


@lru_cache(maxsize=300)
def get_game_team_player_status(
    game_id: str,
    team_abbr: str,
    player_id: str
):
    """
    Consulta o roster específico da partida. A API Core da ESPN retorna
    a coleção de atletas em "entries" (não em "athletes"/"items") e pode
    entregar referências $ref que precisam ser resolvidas.
    """
    summary = get_game_summary(game_id)
    competitions = (
        summary.get("header", {})
        .get("competitions", [])
    )

    team_id = None
    for competition in competitions:
        for competitor in competition.get("competitors", []):
            competitor_team = (
                competitor.get("team", {}).get("abbreviation", "")
            ).upper()
            if competitor_team == str(team_abbr or "").upper():
                team_id = str(
                    competitor.get("id")
                    or competitor.get("team", {}).get("id")
                    or ""
                ).strip()
                break
        if team_id:
            break

    if not team_id:
        return None

    roster_url = (
        "https://sports.core.api.espn.com/v2/sports/basketball/"
        "leagues/nba/events/{event}/competitions/{competition}/"
        "competitors/{team}/roster"
    ).format(
        event=str(game_id),
        competition=str(game_id),
        team=team_id
    )
    roster = request_json(roster_url)

    # O endpoint Core usa "entries". Mantém compatibilidade com formatos
    # alternativos, mas prioriza a estrutura realmente retornada pela ESPN.
    entries = roster.get("entries")
    if not isinstance(entries, list):
        entries = roster.get("athletes")
    if not isinstance(entries, list):
        entries = roster.get("items")
    if not isinstance(entries, list):
        return None

    def resolve_reference(value):
        if isinstance(value, str) and value.startswith("http"):
            return request_json(value)
        if isinstance(value, dict) and value.get("$ref"):
            return request_json(value["$ref"])
        return value if isinstance(value, dict) else {}

    target_id = str(player_id).strip()

    for raw_entry in entries:
        entry = resolve_reference(raw_entry)
        if not isinstance(entry, dict):
            continue

        athlete_value = entry.get("athlete")
        athlete = resolve_reference(athlete_value)
        if not athlete and entry.get("player"):
            athlete = resolve_reference(entry.get("player"))
        athlete_id = str(
            athlete.get("id")
            or entry.get("playerId")
            or entry.get("athleteId")
            or entry.get("id")            or ""
        ).strip()

        # Algumas respostas guardam a referência do atleta em um objeto
        # com URL e o ID só aparece após resolver esse recurso.
        if athlete_id != target_id:
            continue

        did_not_play = (
            entry.get("didNotPlay")
            or entry.get("did_not_play")
            or entry.get("notPlayed")
            or athlete.get("didNotPlay")
            or athlete.get("did_not_play")
        )
        active = entry.get("active")
        if active is None:
            active = athlete.get("active")

        return {
            "player_id": athlete_id,
            "did_not_play": bool(did_not_play),
            "active": active,
            "reason": (
                entry.get("reason")
                or entry.get("notPlayingReason")
                or entry.get("status", {}).get("description")
                or athlete.get("reason")
            )
        }

    return None


def extract_player_game_from_summary(
    summary,
    player_id,
    player_name=None,
    team_abbr=None
):

    player_id = str(
        player_id
    ).strip()

    players_blocks = (
        summary.get("boxscore", {})
        .get("players", [])
    )

    if not isinstance(
        players_blocks,
        list
    ):
        return None

    for team_block in players_blocks:

        if not isinstance(
            team_block,
            dict
        ):
            continue

        team_abbr = (
            team_block.get("team", {})
            .get("abbreviation")
            or ""
        ).upper()

        statistics_blocks = (
            team_block.get(
                "statistics",
                []
            )
        )

        if isinstance(
            statistics_blocks,
            dict
        ):
            statistics_blocks = [
                statistics_blocks
            ]

        if not isinstance(
            statistics_blocks,
            list
        ):
            continue

        for statistics_block in statistics_blocks:

            if not isinstance(
                statistics_block,
                dict
            ):
                continue

            keys = (
                statistics_block.get("keys")
                or statistics_block.get("names")
                or statistics_block.get("labels")
                or []
            )

            key_indexes = {}

            for index, key in enumerate(keys):

                normalized_key = normalize_stat_key(
                    key
                )

                if normalized_key:
                    key_indexes[
                        normalized_key
                    ] = index

            athletes = statistics_block.get(
                "athletes",
                []
            )

            if not isinstance(
                athletes,
                list
            ):
                continue

            for athlete_entry in athletes:

                if not isinstance(
                    athlete_entry,
                    dict
                ):
                    continue

                athlete = athlete_entry.get(
                    "athlete",
                    {}
                )

                if not isinstance(
                    athlete,
                    dict
                ):
                    continue

                athlete_espn_id = str(
                    athlete.get("id", "")
                ).strip()

                if (
                    not athlete_espn_id
                    or athlete_espn_id != player_id
                ):
                    # Em snapshots antigos, o ID do atleta pode não coincidir
                    # com o ID atualmente entregue pelo boxscore da ESPN.
                    # Fallback seguro: mesmo time + nome normalizado.
                    snapshot_name = normalize_diagnostic_player_name(player_name)
                    boxscore_name = normalize_diagnostic_player_name(
                        athlete.get("displayName")
                        or athlete.get("fullName")
                        or athlete.get("shortName")
                        or ""
                    )
                    boxscore_team = team_abbr.upper()
                    expected_team = str(team_abbr or "").upper()
                    if (
                        snapshot_name
                        and snapshot_name == boxscore_name
                        and expected_team
                        and expected_team == boxscore_team
                    ):
                        logger.info(
                            "Audit boxscore player matched by name fallback: player=%s snapshot_id=%s boxscore_id=%s team=%s",
                            player_name,
                            player_id,
                            athlete_espn_id,
                            boxscore_team
                        )
                    else:
                        continue

                player_name = (
                    athlete.get("displayName")
                    or athlete.get("fullName")
                    or athlete.get("shortName")
                    or ""
                )

                raw_stats = (
                    athlete_entry.get("stats")
                    or athlete_entry.get(
                        "statistics"
                    )
                )

                # A ESPN pode listar atletas que não participaram da partida
                # sem bloco de estatísticas. Se o próprio registro do boxscore
                # marcar didNotPlay, esse caso é DNP confirmado, não pendência.
                did_not_play = (
                    athlete_entry.get("didNotPlay")
                    or athlete_entry.get("did_not_play")
                )
                dnp_reason = athlete_entry.get("reason")

                if raw_stats is None:
                    if did_not_play:
                        return {
                            "player_id": player_id,
                            "player": player_name,
                            "team": team_abbr,
                            "minutes": 0.0,
                            "assists": None,
                            "rebounds": None,
                            "points": None,
                            "fga": None,
                            "three_pa": None,
                            "two_pa": None,
                            "fgm": None,
                            "did_not_play": True,
                            "dnp_reason": dnp_reason
                        }
                    return None

                minutes = get_player_stat_value(
                    raw_stats,
                    key_indexes,
                    "minutes"
                )

                assists = get_player_stat_value(
                    raw_stats,
                    key_indexes,
                    "assists"
                )

                rebounds = get_player_stat_value(
                    raw_stats,
                    key_indexes,
                    "rebounds"
                )

                points = get_player_stat_value(
                    raw_stats,
                    key_indexes,
                    "points"
                )

                fga = get_player_stat_value(
                    raw_stats,
                    key_indexes,
                    "fieldGoalsMade-fieldGoalsAttempted",
                    attempt_stat=True
                )

                three_pa = get_player_stat_value(
                    raw_stats,
                    key_indexes,
                    "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                    attempt_stat=True
                )
                two_pa = subtract_stats(fga, three_pa)
                fgm = get_player_stat_value(
                    raw_stats,
                    key_indexes,
                    "fieldGoalsMade-fieldGoalsAttempted",
                    made_stat=True
                )

                return {
                    "player_id": player_id,
                    "player": player_name,
                    "team": team_abbr,
                    "minutes": minutes,
                    "assists": assists,
                    "rebounds": rebounds,
                    "points": points,
                    "fga": fga,
                    "three_pa": three_pa,
                    "two_pa": two_pa,
                    "fgm": fgm
                }

    return None


@lru_cache(maxsize=60)
def get_team_regular_season_schedule(
    team_abbr: str,
    year: int
) -> tuple:

    schedule = request_json(
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"basketball/nba/teams/{team_abbr.lower()}/schedule",
        {"season": year}
    )

    games = get_completed_team_games(
        schedule,
        team_abbr.upper(),
        1000
    )

    return tuple(games)


NBA_TEAM_ABBREVIATIONS = (
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"
)


@lru_cache(maxsize=2)
def get_league_position_defense_baseline(year, target_date_iso):
    """Média da liga de totais cedidos por jogo a cada posição, usando os últimos 5/10 jogos de cada equipe."""
    started = time.perf_counter()
    team_samples = {}

    def fetch_schedule(team):
        games = [
            game for game in get_team_regular_season_schedule(team, year)
            if (game.get("date") or "")[:10] < target_date_iso
        ]
        games.sort(key=lambda game: game.get("date") or "", reverse=True)
        return team, games[:10]

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch_schedule, team) for team in NBA_TEAM_ABBREVIATIONS]
        for future in as_completed(futures):
            try:
                team, games = future.result()
                team_samples[team] = games
            except Exception as error:
                logger.warning("Falha ao consultar agenda da liga: %s", error)

    game_ids = {
        str(game["game_id"])
        for games in team_samples.values()
        for game in games
        if game.get("game_id")
    }

    summaries = {}
    player_status_by_game = {}

    def fetch_summary(game_id):
        return game_id, get_game_summary(game_id)

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch_summary, game_id) for game_id in game_ids]
        for future in as_completed(futures):
            try:
                game_id, summary = future.result()
                summaries[game_id] = summary
            except Exception as error:
                logger.warning("Falha ao consultar jogo para benchmark da liga: %s", error)

    # Aproveita os mesmos boxscores já carregados pelo benchmark defensivo.
    # Não dispara nenhuma chamada adicional à ESPN para o contexto de companheiros.
    for game_id, summary in summaries.items():
        status_index = {}
        player_blocks = (summary.get("boxscore") or {}).get("players") or []
        team_abbreviations = {
            str((block.get("team") or {}).get("abbreviation") or "").upper()
            for block in player_blocks
            if (block.get("team") or {}).get("abbreviation")
        }
        for team_abbr in team_abbreviations:
            status_index.update(
                extract_team_player_status(summary, team_abbr)
            )
        player_status_by_game[str(game_id)] = status_index

    stat_keys = ("points", "assists", "rebounds", "fga", "three_pa", "two_pa", "fgm")
    samples = {
        window: {
            position: {stat: [] for stat in stat_keys}
            for position in ("G", "F", "C")
        }
        for window in (5, 10)
    }
    # Guarda somente os dados individuais já extraídos dos mesmos boxscores.
    # Isso permite comparar a faixa de minutos sem novas consultas à ESPN.
    minute_player_samples = {
        window: {
            position: []
            for position in ("G", "F", "C")
        }
        for window in (5, 10)
    }

    for team, games in team_samples.items():
        for index, game in enumerate(games[:10]):
            summary = summaries.get(str(game.get("game_id")))
            if not summary:
                continue
            try:
                positions = aggregate_position_game(summary, team).get("positions", {})
            except Exception as error:
                logger.warning("Falha ao agregar benchmark %s/%s: %s", team, game.get("game_id"), error)
                continue

            for position in ("G", "F", "C"):
                position_data = positions.get(position, {})
                players = position_data.get("players", [])
                if players:
                    sample_game = {
                        "positions": {
                            position: {
                                "players": players
                            }
                        }
                    }
                    minute_player_samples[10][position].append(sample_game)
                    if index < 5:
                        minute_player_samples[5][position].append(sample_game)

                for stat_key in stat_keys:
                    value = position_data.get(stat_key)
                    if value is None:
                        continue
                    samples[10][position][stat_key].append(value)
                    if index < 5:
                        samples[5][position][stat_key].append(value)

    result = {
        "positions": {},
        "minute_player_samples": minute_player_samples,
        "teams_analyzed": len(team_samples),
        "unique_games": len(summaries),
        "player_status_by_game": player_status_by_game
    }
    for position in ("G", "F", "C"):
        result["positions"][position] = {}
        for stat_key in stat_keys:
            result["positions"][position][stat_key] = {}
            for window in (5, 10):
                values = samples[window][position][stat_key]
                expected_games = sum(min(window, len(games)) for games in team_samples.values())
                has_sufficient_coverage = (
                    len(team_samples) >= 24
                    and expected_games > 0
                    and len(values) >= expected_games * 0.70
                )
                result["positions"][position][stat_key][f"l{window}_avg"] = (
                    round(sum(values) / len(values), 2)
                    if values and has_sufficient_coverage else None
                )
                result["positions"][position][stat_key][f"l{window}_games"] = len(values)

    logger.info(
        "Benchmark liga: data=%s equipes=%s jogos_unicos=%s L5_G=%s L10_G=%s tempo=%.2fs",
        target_date_iso, len(team_samples), len(summaries),
        result["positions"]["G"]["assists"]["l5_games"],
        result["positions"]["G"]["assists"]["l10_games"],
        time.perf_counter() - started
    )
    return result


# Serializa a construção do benchmark por processo. Isso evita que o
# prewarm e a primeira consulta de candidatos montem o mesmo benchmark em paralelo.
_LEAGUE_BASELINE_SINGLEFLIGHT_LOCK = threading.Lock()


def get_league_position_defense_baseline_singleflight(year, target_date_iso):
    with _LEAGUE_BASELINE_SINGLEFLIGHT_LOCK:
        return get_league_position_defense_baseline(year, target_date_iso)


_DEFENSE_PREWARM_LOCK = threading.Lock()
_DEFENSE_PREWARM_DATES = set()
_DEFENSE_PREWARM_STATE = {}


def schedule_defense_context_prewarm(selected_date):
    """Prepara o benchmark defensivo da data sem bloquear /games."""
    with _DEFENSE_PREWARM_LOCK:
        if selected_date in _DEFENSE_PREWARM_DATES:
            return
        _DEFENSE_PREWARM_DATES.add(selected_date)
        _DEFENSE_PREWARM_STATE[selected_date] = {
            "status": "running",
            "started_at": time.perf_counter(),
            "elapsed": None
        }

    def warm():
        started = time.perf_counter()        try:
            target_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
            baseline = get_league_position_defense_baseline_singleflight(
                season_year(target_date),
                selected_date
            )
            elapsed = time.perf_counter() - started            with _DEFENSE_PREWARM_LOCK:
                _DEFENSE_PREWARM_STATE[selected_date] = {
                    "status": "ready",
                    "started_at": started,
                    "elapsed": elapsed
                }
            logger.info(
                "Defense prewarm: data=%s status=ok equipes=%s jogos_unicos=%s tempo=%.2fs",
                selected_date,
                baseline.get("teams_analyzed", 0),
                baseline.get("unique_games", 0),
                elapsed
            )
        except Exception as error:
            elapsed = time.perf_counter() - started
            with _DEFENSE_PREWARM_LOCK:
                _DEFENSE_PREWARM_DATES.discard(selected_date)
                _DEFENSE_PREWARM_STATE[selected_date] = {
                    "status": "error",
                    "started_at": started,
                    "elapsed": elapsed,
                    "error": str(error)
                }
            logger.warning(
                "Defense prewarm: data=%s status=error tempo=%.2fs erro=%s",
                selected_date,
                time.perf_counter() - started,
                error
            )

    threading.Thread(
        target=warm,
        name=f"defense-prewarm-{selected_date}",
        daemon=True
    ).start()


def get_player_team_game_counts(
    player_games,
    team_abbr: str,
    target_date: date,
    year: int
) -> dict:

    team_schedule = [
        game
        for game in get_team_regular_season_schedule(
            team_abbr,
            year
        )
        if (
            game.get("date") or ""
        )[:10] < target_date.isoformat()
    ]

    team_schedule.sort(
        key=lambda game: game.get("date") or "",
        reverse=True
    )

    player_appearances = {
        str(game.get("event_id")): game
        for game in player_games
        if (
            game.get("event_id")
            and parse_minutes(
                game.get("minutes")
            ) > 0
        )
    }

    result = {}

    for window in (5, 10):

        sample = team_schedule[:window]

        played = sum(
            1
            for team_game in sample
            if str(
                team_game.get("game_id")
            ) in player_appearances
        )

        result[
            f"team_games_last{window}_played"
        ] = played

        result[
            f"team_games_last{window}_total"
        ] = len(sample)

        result[
            f"team_games_last{window}_pct"
        ] = (
            round(
                played / len(sample) * 100,
                1
            )
            if sample
            else None
        )

    return result


def get_opponents_for_date(
    schedule
):

    opponents = {}

    for game in schedule.get(
        "games",
        []
    ):

        home = game["home"]["abbr"]
        away = game["away"]["abbr"]
        opponents[home] = away
        opponents[away] = home

    return opponents


def build_player_venue_production(games, target_date, stat_key):
    """
    Produção individual por mando de quadra.
    L10 é independente por local; a temporada considera todos os jogos
    anteriores à data-alvo com homeAway válido.
    """
    previous_games = [
        game for game in games
        if game.get("date") and game["date"] < target_date.isoformat()
    ]

    result = {}
    for venue in ("home", "away"):
        venue_games = [
            game for game in previous_games
            if game.get("home_away") == venue
            and game.get(stat_key) is not None
        ]
        last10 = venue_games[:10]

        def summarize(sample):
            if not sample:
                return {
                    "average": None,
                    "games": 0,
                    "minutes_avg": None
                }
            return {
                "average": round(
                    sum(float(game[stat_key]) for game in sample) / len(sample), 2
                ),
                "games": len(sample),
                "minutes_avg": round(
                    sum(float(game.get("minutes") or 0) for game in sample) / len(sample), 1
                )
            }

        result[venue] = {
            "season": summarize(venue_games),
            "last10": summarize(last10)
        }

    return result


def build_player_result(
    player,
    games,
    target_date,
    market,
    defense=None,
    opponent=None,
    league_baseline=None
):

    previous_games = [
        game
        for game in games
        if game["date"] < target_date.isoformat()
    ]

    if len(previous_games) < 3:
        return None

    previous_games = previous_games[:10]
    last5 = previous_games[:5]
    last10 = previous_games[:10]

    minutes_avg = (
        sum(
            game["minutes"]
            for game in last5
        )
        / len(last5)
    )

    if minutes_avg < 24:
        return None

    stat_key = {
        "points": "points",
        "assists": "assists",
        "rebounds": "rebounds",
        "threes": "three_pa",
        "two_pa": "two_pa",
        "fgm": "fgm",
        "attempts": "fga"
    }[market]

    defense_key = {
        "points": "fga_allowed",
        "assists": "assists_allowed",
        "rebounds": "rebounds_allowed",
        "threes": "three_pa_allowed",
        "two_pa": "two_pa_allowed",
        "fgm": "fgm_allowed",
        "attempts": "fga_allowed"
    }[market]

    def average(sample):

        if not sample:
            return 0.0

        return sum(
            game[stat_key]
            for game in sample
        ) / len(sample)

    avg5 = average(last5)
    avg10 = average(last10)

    # Frequência de jogos L5 que atingem/superam a própria média L5.
    # Este fator não mede dispersão ou estabilidade estatística.
    above_l5_mean_rate = (
        sum(
            game[stat_key] >= avg5
            for game in last5
        )
        / len(last5)
    )

    score = (
        avg5 * 0.60
        + avg10 * 0.40
    ) * (
        0.80
        + above_l5_mean_rate * 0.20
    )

    defense_l5 = (
        defense.get("L5", {})
        .get("averages", {})
        .get(defense_key)
        if defense
        else None
    )

    defense_l10 = (
        defense.get("L10", {})
        .get("averages", {})
        .get(defense_key)
        if defense
        else None
    )

    defense_games_l5 = (
        defense.get("L5", {})
        .get("games_analyzed", 0)
        if defense
        else 0
    )

    defense_games_l10 = (
        defense.get("L10", {})
        .get("games_analyzed", 0)
        if defense
        else 0
    )

    position = normalize_position(
        player.get("position")
    )

    position_stat_key = {
        "points": "points",
        "assists": "assists",
        "rebounds": "rebounds",
        "threes": "three_pa",
        "two_pa": "two_pa",
        "fgm": "fgm",
        "attempts": "fga"
    }.get(market)

    position_defense_l5 = None
    position_defense_l10 = None
    position_individual_l5 = None
    position_individual_l10 = None
    position_games_l5 = 0
    position_games_l10 = 0
    position_player_games_l5 = 0
    position_player_games_l10 = 0
    position_filtered_l5 = None
    position_filtered_l10 = None
    position_minute_band_l5 = None
    position_minute_band_l10 = None
    league_minute_band_l5 = None
    league_minute_band_l10 = None
    position_equal_game_l5 = None
    position_equal_game_l10 = None

    if (
        defense
        and position in ("G", "F", "C")
        and position_stat_key
    ):

        position_data_l5 = (
            defense.get("L5", {})
            .get("positions", {})
            .get(position, {})
        )

        position_data_l10 = (
            defense.get("L10", {})
            .get("positions", {})
            .get(position, {})
        )

        position_defense_l5 = (
            position_data_l5
            .get("averages", {})
            .get(position_stat_key)
        )

        position_defense_l10 = (
            position_data_l10
            .get("averages", {})
            .get(position_stat_key)
        )

        position_games_l5 = (
            position_data_l5
            .get("stat_games_analyzed", {})
            .get(position_stat_key, 0)
        )

        position_games_l10 = (
            position_data_l10
            .get("stat_games_analyzed", {})
            .get(position_stat_key, 0)
        )

        position_individual_l5 = (
            position_data_l5
            .get("individual_averages", {})
            .get(position_stat_key)
        )

        position_individual_l10 = (
            position_data_l10
            .get("individual_averages", {})
            .get(position_stat_key)
        )

        position_player_games_l5 = (
            position_data_l5
            .get("stat_player_appearances", {})
            .get(position_stat_key, 0)
        )

        position_player_games_l10 = (
            position_data_l10
            .get("stat_player_appearances", {})
            .get(position_stat_key, 0)
        )

        defense_games_sample_l5 = (
            defense.get("L5", {})
            .get("games", [])
        )

        defense_games_sample_l10 = (
            defense.get("L10", {})
            .get("games", [])
        )

        position_filtered_l5 = (
            calculate_minutes_filtered_position_defense(
                defense_games_sample_l5,
                position,
                position_stat_key,
                minutes_avg
            )
        )

        position_filtered_l10 = (
            calculate_minutes_filtered_position_defense(
                defense_games_sample_l10,
                position,
                position_stat_key,
                minutes_avg
            )
        )

        # Nova leitura contextual: jogadores da mesma posição em uma faixa
        # simétrica de ±5 minutos do candidato, usando os mesmos boxscores.
        position_minute_band_l5 = calculate_minutes_band_position_defense(
            defense_games_sample_l5,
            position,
            position_stat_key,
            minutes_avg,
            5.0
        )
        position_minute_band_l10 = calculate_minutes_band_position_defense(
            defense_games_sample_l10,
            position,
            position_stat_key,
            minutes_avg,
            5.0
        )

        league_minute_samples = (
            league_baseline.get("minute_player_samples", {})
            if league_baseline else {}
        )
        league_minute_band_l5 = calculate_minutes_band_position_defense(
            league_minute_samples.get(5, {}).get(position, []),
            position,
            position_stat_key,
            minutes_avg,
            5.0
        )
        league_minute_band_l10 = calculate_minutes_band_position_defense(
            league_minute_samples.get(10, {}).get(position, []),
            position,
            position_stat_key,
            minutes_avg,
            5.0
        )

        position_equal_game_l5 = calculate_equal_game_position_defense(
            defense_games_sample_l5, position, position_stat_key
        )
        position_equal_game_l10 = calculate_equal_game_position_defense(
            defense_games_sample_l10, position, position_stat_key
        )

    team_game_counts = get_player_team_game_counts(
        games,
        player["team"],
        target_date,
        season_year(target_date)
    )

    return {
        "player_id": str(player["id"]),
        "player": player["player"],
        "team": player["team"],
        "position": position,
        "opponent": opponent,
        "last_game_team": last5[0].get("team"),
        "team_changed": bool(
            last5[0].get("team")
            and last5[0]["team"] != player["team"]
        ),
        "last5_avg": round(avg5, 2),
        "last10_avg": round(avg10, 2),
        "minutes_last5_avg": round(minutes_avg, 1),
        "venue_production": build_player_venue_production(
            games, target_date, stat_key
        ),
        "triage_score": round(score, 2),
        "games_last5": len(last5),
        "games_last10": len(last10),
        **team_game_counts,
        "opponent_defense_l5_avg": defense_l5,
        "opponent_defense_l10_avg": defense_l10,
        "opponent_defense_games_l5": defense_games_l5,
        "opponent_defense_games_l10": defense_games_l10,
        "opponent_position_defense_l5_avg": position_defense_l5,
        "opponent_position_defense_l10_avg": position_defense_l10,
        "opponent_position_defense_games_l5": position_games_l5,
        "opponent_position_defense_games_l10": position_games_l10,
        "opponent_position_individual_defense_l5_avg": position_individual_l5,
        "opponent_position_individual_defense_l10_avg": position_individual_l10,
        "opponent_position_individual_defense_player_games_l5": position_player_games_l5,        "opponent_position_individual_defense_player_games_l10": position_player_games_l10,
        "opponent_position_minutes_filtered_threshold": round(minutes_avg, 1),
        "opponent_position_minute_band_reference_version": 1,
        "opponent_position_minute_band_margin": 5.0,
        "opponent_position_minute_band_center": round(minutes_avg, 1),
        "opponent_position_minute_band_lower": (
            position_minute_band_l5.get("minutes_lower")
            if position_minute_band_l5 else None        ),
        "opponent_position_minute_band_upper": (
            position_minute_band_l5.get("minutes_upper")
            if position_minute_band_l5 else None
        ),
        "opponent_position_minute_band_defense_l5_avg": (
            position_minute_band_l5.get("average_per_player_appearance")
            if position_minute_band_l5 else None
        ),
        "opponent_position_minute_band_defense_l10_avg": (
            position_minute_band_l10.get("average_per_player_appearance")
            if position_minute_band_l10 else None
        ),
        "opponent_position_minute_band_defense_l5_games": (
            position_minute_band_l5.get("games_analyzed", 0)
            if position_minute_band_l5 else 0
        ),
        "opponent_position_minute_band_defense_l10_games": (
            position_minute_band_l10.get("games_analyzed", 0)
            if position_minute_band_l10 else 0
        ),
        "opponent_position_minute_band_defense_l5_appearances": (
            position_minute_band_l5.get("player_appearances", 0)
            if position_minute_band_l5 else 0
        ),
        "opponent_position_minute_band_defense_l10_appearances": (
            position_minute_band_l10.get("player_appearances", 0)
            if position_minute_band_l10 else 0
        ),
        "league_position_minute_band_l5_avg": (
            league_minute_band_l5.get("average_per_player_appearance")
            if league_minute_band_l5 else None
        ),
        "league_position_minute_band_l10_avg": (
            league_minute_band_l10.get("average_per_player_appearance")
            if league_minute_band_l10 else None
        ),
        "league_position_minute_band_l5_games": (
            league_minute_band_l5.get("games_analyzed", 0)
            if league_minute_band_l5 else 0
        ),
        "league_position_minute_band_l10_games": (
            league_minute_band_l10.get("games_analyzed", 0)
            if league_minute_band_l10 else 0
        ),
        "league_position_minute_band_l5_appearances": (
            league_minute_band_l5.get("player_appearances", 0)
            if league_minute_band_l5 else 0
        ),
        "league_position_minute_band_l10_appearances": (
            league_minute_band_l10.get("player_appearances", 0)
            if league_minute_band_l10 else 0
        ),
        "opponent_position_minutes_filtered_defense_l5_avg": (
            position_filtered_l5.get("average_per_player_appearance")
            if position_filtered_l5 else None
        ),
        "opponent_position_minutes_filtered_defense_l10_avg": (
            position_filtered_l10.get("average_per_player_appearance")
            if position_filtered_l10 else None
        ),
        "opponent_position_minutes_filtered_defense_l5_per_minute": (
            position_filtered_l5.get("per_minute")
            if position_filtered_l5 else None
        ),
        "opponent_position_minutes_filtered_defense_l10_per_minute": (
            position_filtered_l10.get("per_minute")
            if position_filtered_l10 else None
        ),
        "opponent_position_minutes_filtered_defense_l5_projected": (
            position_filtered_l5.get("projected_at_player_minutes")
            if position_filtered_l5 else None
        ),
        "opponent_position_minutes_filtered_defense_l10_projected": (
            position_filtered_l10.get("projected_at_player_minutes")
            if position_filtered_l10 else None
        ),
        "opponent_position_minutes_filtered_defense_games_l5": (
            position_filtered_l5.get("games_analyzed", 0)
            if position_filtered_l5 else 0
        ),
        "opponent_position_minutes_filtered_defense_games_l10": (
            position_filtered_l10.get("games_analyzed", 0)
            if position_filtered_l10 else 0
        ),
        "opponent_position_minutes_filtered_defense_player_games_l5": (
            position_filtered_l5.get("player_appearances", 0)
            if position_filtered_l5 else 0
        ),
        "opponent_position_minutes_filtered_defense_player_games_l10": (
            position_filtered_l10.get("player_appearances", 0)
            if position_filtered_l10 else 0
        ),
        "opponent_position_minutes_filtered_defense_minutes_l5": (
            position_filtered_l5.get("total_minutes", 0)
            if position_filtered_l5 else 0
        ),
        "opponent_position_minutes_filtered_defense_minutes_l10": (
            position_filtered_l10.get("total_minutes", 0)
            if position_filtered_l10 else 0
        ),
        "opponent_position_equal_game_defense_l5_avg": position_equal_game_l5.get("average_per_game") if position_equal_game_l5 else None,
        "opponent_position_equal_game_defense_l10_avg": position_equal_game_l10.get("average_per_game") if position_equal_game_l10 else None,
        "opponent_position_equal_game_defense_l5_games": position_equal_game_l5.get("games_analyzed", 0) if position_equal_game_l5 else 0,
        "opponent_position_equal_game_defense_l10_games": position_equal_game_l10.get("games_analyzed", 0) if position_equal_game_l10 else 0,
        "opponent_position_equal_game_defense_l5_appearances": position_equal_game_l5.get("player_appearances", 0) if position_equal_game_l5 else 0,
        "opponent_position_equal_game_defense_l10_appearances": position_equal_game_l10.get("player_appearances", 0) if position_equal_game_l10 else 0,
        "opponent_position_allowed_l5_avg": position_defense_l5,
        "opponent_position_allowed_l10_avg": position_defense_l10,
        "league_position_allowed_l5_avg": (
            league_baseline.get("positions", {}).get(position, {}).get(position_stat_key, {}).get("l5_avg")
            if league_baseline and position_stat_key else None
        ),
        "league_position_allowed_l10_avg": (
            league_baseline.get("positions", {}).get(position, {}).get(position_stat_key, {}).get("l10_avg")
            if league_baseline and position_stat_key else None
        ),
        "league_position_allowed_l5_games": (
            league_baseline.get("positions", {}).get(position, {}).get(position_stat_key, {}).get("l5_games", 0)
            if league_baseline and position_stat_key else 0
        ),
        "league_position_allowed_l10_games": (
            league_baseline.get("positions", {}).get(position, {}).get(position_stat_key, {}).get("l10_games", 0)
            if league_baseline and position_stat_key else 0
        ),
        "opponent_position_equal_game_reference_version": 2
    }


def collect_candidates(
    selected_date,
    market,
    limit,
    persist_operational=True
):

    total_start = time.perf_counter()

    target_date = datetime.strptime(
        selected_date,
        "%Y-%m-%d"
    ).date()

    year = season_year(target_date)

    stage_start = time.perf_counter()

    roster_data = get_players_for_date(
        selected_date
    )

    roster_elapsed = time.perf_counter() - stage_start

    players = roster_data["players"]

    # Relatório atual da ESPN usado somente para elegibilidade pré-jogo.
    # Não altera triage_score nem a fotografia imutável dos candidatos.
    availability_start = time.perf_counter()
    availability_report_status = "ok"
    availability_report_warning = None
    try:
        injury_payload = request_json(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        )
        injury_index = parse_espn_injuries(injury_payload)
        if not injury_index:
            availability_report_status = "empty"
            availability_report_warning = "A ESPN retornou um relatório válido, mas sem atletas listados."
            logger.warning("Relatório de lesões ESPN vazio; nenhum atleta será presumido ativo.")
    except Exception as error:
        injury_index = {}
        availability_report_status = "unavailable"
        availability_report_warning = "Não foi possível validar o relatório de lesões; confira a disponibilidade manualmente."
        logger.warning("Falha ao consultar disponibilidade ESPN: %s", error)

    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=disponibilidade status=%s reportados=%s tempo=%.2fs",
        selected_date,
        market,
        availability_report_status,
        len(injury_index),
        time.perf_counter() - availability_start
    )

    opponents = get_opponents_for_date(
        roster_data["schedule"]
    )

    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=rosters jogadores=%s tempo=%.2fs",
        selected_date,
        market,
        len(players),
        roster_elapsed
    )

    # O histórico individual pode ter sido aquecido em segundo plano pelo
    # endpoint /games. Se ainda estiver em andamento, esperamos o trabalho
    # existente em vez de disparar uma segunda bateria de 317 chamadas.

    wait_player_history_prewarm(selected_date)

    # Primeiro buscamos os históricos dos jogadores. Isso permite
    # identificar quem realmente pode virar candidato antes de
    # consultar o histórico defensivo dos adversários.
    stage_start = time.perf_counter()
    player_logs = {}
    successful_logs = 0
    errors = 0
    # Registra se o prewarm iniciado pela busca de jogos terminou antes
    # da primeira consulta de mercado. Não espera nem altera o fluxo do Scanner.
    with _PLAYER_HISTORY_PREWARM_LOCK:
        player_prewarm_state = dict(
            _PLAYER_HISTORY_PREWARM_STATE.get(selected_date, {})
        )
    logger.info(
        "Player history prewarm at candidates start: data=%s status=%s "
        "elapsed=%.2fs players=%s successful=%s errors=%s",
        selected_date,
        player_prewarm_state.get("status", "not_started"),
        (
            time.perf_counter() - player_prewarm_state["started"]
            if player_prewarm_state.get("started") is not None
            else 0.0
        ),
        player_prewarm_state.get("players", 0),
        player_prewarm_state.get("successful", 0),
        player_prewarm_state.get("errors", 0),
    )

    player_cache_before = _cache_metrics(get_athlete_logs)

    player_fetch_durations = []
    player_http_durations = []
    player_fetch_errors = 0

    def fetch_player(player):
        fetch_started = time.perf_counter()
        _http_local.last_gamelog_http_elapsed = None
        try:
            logs = get_athlete_logs(
                player["id"],
                year
            )
            error = None
        except Exception as caught_error:
            logs = None
            error = caught_error

        fetch_elapsed = time.perf_counter() - fetch_started
        http_elapsed = getattr(
            _http_local,
            "last_gamelog_http_elapsed",
            None
        )
        return player, logs, fetch_elapsed, http_elapsed, error

    # Concorrência controlada no histórico individual.
    # O limite de 16 foi o melhor resultado medido até aqui; aumentos
    # adicionais serão tratados somente após otimização estrutural.
    player_history_workers = 16
    with ThreadPoolExecutor(
        max_workers=player_history_workers
    ) as executor:

        futures = [
            executor.submit(
                fetch_player,
                player
            )
            for player in players
        ]

        for future in as_completed(futures):

            try:
                player, logs, fetch_elapsed, http_elapsed, fetch_error = future.result()
                player_fetch_durations.append(fetch_elapsed)
                if http_elapsed is not None:
                    player_http_durations.append(http_elapsed)

                if fetch_error is not None:
                    errors += 1
                    player_fetch_errors += 1
                    logger.warning(
                        "Falha ao consultar jogador: %s",
                        fetch_error
                    )
                    continue

                successful_logs += 1
                player_logs[player["id"]] = (player, logs)

            except Exception as error:
                errors += 1
                player_fetch_errors += 1
                logger.warning(
                    "Falha ao processar retorno de jogador: %s",
                    error
                )

    player_logs_elapsed = time.perf_counter() - stage_start

    def latency_summary(values):
        if not values:
            return {
                "count": 0,
                "avg": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "max": 0.0
            }

        ordered = sorted(values)
        count = len(ordered)
        middle = count // 2
        median = (
            ordered[middle]
            if count % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        p95_index = min(count - 1, int((count - 1) * 0.95))
        return {
            "count": count,
            "avg": sum(ordered) / count,
            "median": median,
            "p95": ordered[p95_index],
            "max": ordered[-1]
        }

    fetch_latency = latency_summary(player_fetch_durations)
    http_latency = latency_summary(player_http_durations)
    logger.info(
        "Candidatos player latency: data=%s mercado=%s workers=%s "
        "fetch_count=%s fetch_avg=%.3fs fetch_median=%.3fs fetch_p95=%.3fs fetch_max=%.3fs "
        "http_count=%s http_avg=%.3fs http_median=%.3fs http_p95=%.3fs http_max=%.3fs erros=%s",
        selected_date,
        market,
        player_history_workers,
        fetch_latency["count"],
        fetch_latency["avg"],
        fetch_latency["median"],
        fetch_latency["p95"],
        fetch_latency["max"],
        http_latency["count"],
        http_latency["avg"],
        http_latency["median"],
        http_latency["p95"],
        http_latency["max"],
        player_fetch_errors
    )

    # Mantém exatamente os mesmos critérios de elegibilidade usados
    # em build_player_result(): mínimo de 3 jogos anteriores e média
    # de minutos L5 >= 24. Nenhuma fórmula de score é alterada aqui.
    eligible_players = []

    for player, logs in player_logs.values():

        previous_games = [
            game
            for game in logs
            if game["date"] < target_date.isoformat()
        ]

        if len(previous_games) < 3:
            continue

        last5 = previous_games[:5]

        minutes_avg = (
            sum(
                game["minutes"]
                for game in last5
            )
            / len(last5)
        )

        if minutes_avg < 24:
            continue

        eligible_players.append(
            (player, logs)
        )

    player_cache_after = _cache_metrics(get_athlete_logs)
    logger.info(
        "Candidatos cache: data=%s mercado=%s cache=get_athlete_logs hits_delta=%s misses_delta=%s size=%s",
        selected_date,
        market,
        player_cache_after["hits"] - player_cache_before["hits"],
        player_cache_after["misses"] - player_cache_before["misses"],
        player_cache_after["currsize"]
    )
    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=historico_jogadores jogadores_consultados=%s elegiveis=%s tempo=%.2fs",
        selected_date,
        market,
        successful_logs,
        len(eligible_players),
        player_logs_elapsed
    )

    # Só consultamos defesa para adversários que realmente possuem
    # pelo menos um jogador elegível para este mercado/data.
    stage_start = time.perf_counter()
    defense_cache_before = _cache_metrics(get_team_defense_history)
    schedule_cache_before = _cache_metrics(get_team_regular_season_schedule)
    summary_cache_before = _cache_metrics(_get_cached_game_summary)
    league_cache_before = _cache_metrics(get_league_position_defense_baseline)

    eligible_opponents = {
        opponents.get(player["team"])
        for player, _ in eligible_players
        if opponents.get(player["team"])
    }

    defense_cache = {}

    def fetch_team_defense(team):
        team_start = time.perf_counter()
        defense = get_team_defense_history(
            team,
            10,
            target_date
        )
        elapsed = time.perf_counter() - team_start
        logger.info(
            "Defesa timing: data=%s equipe=%s jogos=%s tempo=%.2fs",
            selected_date,
            team,
            len(defense),
            elapsed
        )
        return team, defense

    with _DEFENSE_PREWARM_LOCK:
        prewarm_state = dict(_DEFENSE_PREWARM_STATE.get(selected_date, {}))
    logger.info(
        "Defense prewarm status at candidates start: data=%s status=%s elapsed=%.2fs",
        selected_date,
        prewarm_state.get("status", "not_started"),
        prewarm_state.get("elapsed") or 0.0
    )

    league_baseline = None

    # Mantém no máximo 12 consultas de boxscore simultâneas:
    # 6 equipes x 2 boxscores internos. O objetivo é reduzir o tempo total
    # sem repetir o pico de concorrência que poderia pressionar a memória.
    with ThreadPoolExecutor(
        max_workers=6
    ) as executor:

        futures = [
            executor.submit(fetch_team_defense, team)
            for team in eligible_opponents
        ]
        league_future = (
            executor.submit(
                get_league_position_defense_baseline_singleflight,
                year,
                target_date.isoformat()
            )
            if eligible_players else None
        )

        for future in as_completed(futures):
            try:
                team, defense = future.result()
                defense_cache[team] = defense
            except Exception as error:
                logger.warning(
                    "Falha ao consultar histórico defensivo: %s",
                    error
                )

        if league_future:
            try:
                baseline_wait_started = time.perf_counter()
                league_baseline = league_future.result()
                logger.info(
                    "Defense benchmark wait: data=%s mercado=%s prewarm_status_at_start=%s wait=%.2fs",
                    selected_date,
                    market,
                    prewarm_state.get("status", "not_started"),
                    time.perf_counter() - baseline_wait_started
                )
            except Exception as error:
                logger.warning(
                    "Falha ao calcular benchmark defensivo da liga: %s",
                    error
                )
    defense_elapsed = time.perf_counter() - stage_start

    defense_cache_after = _cache_metrics(get_team_defense_history)
    schedule_cache_after = _cache_metrics(get_team_regular_season_schedule)
    summary_cache_after = _cache_metrics(_get_cached_game_summary)
    league_cache_after = _cache_metrics(get_league_position_defense_baseline)

    logger.info(
        "Candidatos cache: data=%s mercado=%s cache=get_team_defense_history hits_delta=%s misses_delta=%s size=%s",
        selected_date,        market,
        defense_cache_after["hits"] - defense_cache_before["hits"],
        defense_cache_after["misses"] - defense_cache_before["misses"],
        defense_cache_after["currsize"]
    )
    logger.info(
        "Candidatos cache: data=%s mercado=%s cache=get_team_regular_season_schedule hits_delta=%s misses_delta=%s size=%s",
        selected_date,
        market,
        schedule_cache_after["hits"] - schedule_cache_before["hits"],
        schedule_cache_after["misses"] - schedule_cache_before["misses"],
        schedule_cache_after["currsize"]
    )
    logger.info(
        "Candidatos cache: data=%s mercado=%s cache=get_game_summary hits_delta=%s misses_delta=%s size=%s",
        selected_date,
        market,
        summary_cache_after["hits"] - summary_cache_before["hits"],
        summary_cache_after["misses"] - summary_cache_before["misses"],
        summary_cache_after["currsize"]
    )
    logger.info(
        "Candidatos cache: data=%s mercado=%s cache=league_position_defense_baseline hits_delta=%s misses_delta=%s size=%s",
        selected_date,
        market,
        league_cache_after["hits"] - league_cache_before["hits"],
        league_cache_after["misses"] - league_cache_before["misses"],
        league_cache_after["currsize"]
    )
    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=defesa adversarios=%s tempo=%.2fs",
        selected_date,
        market,
        len(eligible_opponents),
        defense_elapsed
    )

    stage_start = time.perf_counter()
    results = []

    for player, logs in eligible_players:

        opponent = opponents.get(
            player["team"]
        )

        defense = defense_cache.get(
            opponent
        )

        result = build_player_result(
            player,
            logs,
            target_date,
            market,
            defense,
            opponent,
            league_baseline
        )

        if result:
            results.append(result)

    calculation_elapsed = time.perf_counter() - stage_start

    results.sort(
        key=lambda item: item["triage_score"],
        reverse=True
    )

    stage_start = time.perf_counter()

    if persist_operational:
        save_candidate_snapshots(
            selected_date,
            market,
            results
        )

    database_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - total_start

    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=calculo resultados=%s tempo=%.2fs",
        selected_date,
        market,
        len(results),
        calculation_elapsed
    )

    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=banco tempo=%.2fs total=%.2fs",
        selected_date,
        market,
        database_elapsed,
        total_elapsed
    )

    final_limit = min(limit, 10)

    # Remove somente ausências confirmadas depois de salvar o snapshot.
    # Assim o histórico original e o triage_score permanecem intactos.
    (
        displayed_results,
        unavailable_top10_results,
        outside_top10_results,
        eligible_all_results,
    ) = select_eligible_players(
        results,
        injury_index,
        limit=final_limit
    )

    # Captura histórica: o contexto de companheiros precisa existir ANTES
    # do snapshot imutável. Limitamos a captura histórica aos 10 primeiros
    # candidatos do ranking para não transformar a auditoria em uma nova etapa
    # cara para toda a população elegível. O restante continua sendo calculado
    # apenas quando necessário para a resposta operacional.
    audit_context_start = time.perf_counter()
    if not persist_operational and league_baseline and league_baseline.get("player_status_by_game"):
        player_logs_by_id = player_logs
        roster_by_team = {}
        for roster_player in players:
            roster_by_team.setdefault(
                str(roster_player.get("team") or "").upper(), []
            ).append(roster_player)
        candidate_lookup = {
            str(player.get("id")): (player, logs)
            for player, logs in eligible_players
        }
        status_index = league_baseline["player_status_by_game"]

        for result in eligible_all_results[:10]:
            candidate_id = str(result.get("player_id") or "")
            candidate_entry = candidate_lookup.get(candidate_id)
            if not candidate_entry:
                continue
            candidate_player, candidate_logs = candidate_entry
            team = str(candidate_player.get("team") or "").upper()
            team_games = [
                game for game in get_team_regular_season_schedule(team, year)
                if (game.get("date") or "")[:10] < target_date.isoformat()
            ]
            team_games.sort(
                key=lambda game: game.get("date") or "", reverse=True
            )
            team_games = team_games[:10]
            player_logs_by_id[candidate_id] = (candidate_player, candidate_logs)
            result["teammate_context"] = build_candidate_teammate_context(
                candidate_id=candidate_id,
                team=team,
                roster=roster_by_team.get(team, []),
                player_logs_by_id=player_logs_by_id,
                team_games=team_games,
                teammate_status_by_game=status_index
            )

        logger.info(
            "Auditoria timing: data=%s mercado=%s etapa=contexto_companheiros_snapshot candidatos=%s tempo=%.2fs",
            selected_date,
            market,
            sum(1 for item in eligible_all_results[:10] if item.get("teammate_context")),
            time.perf_counter() - audit_context_start
        )

    # A auditoria independente só captura quando chamada explicitamente pelo
    # endpoint /capture-audit-snapshot. O uso normal do Scanner não pode criar
    # nem alterar a fotografia da auditoria.
    audit_snapshots_saved = 0
    if not persist_operational:
        audit_snapshots_saved = save_audit_snapshots(
            selected_date,
            market,
            eligible_all_results,
            availability_report_status
        )

    # O contexto é anexado somente à resposta, depois do snapshot imutável.
    # Assim não altera candidate_snapshots, Auditoria nem triage_score.
    context_start = time.perf_counter()
    if persist_operational and league_baseline and league_baseline.get("player_status_by_game"):
        player_logs_by_id = player_logs
        roster_by_team = {}
        for roster_player in players:
            roster_by_team.setdefault(
                str(roster_player.get("team") or "").upper(), []
            ).append(roster_player)

        candidate_lookup = {
            str(player.get("id")): (player, logs)
            for player, logs in eligible_players
        }
        status_index = league_baseline["player_status_by_game"]

        for result in displayed_results:
            if result.get("teammate_context"):
                continue
            candidate_id = str(result.get("player_id") or "")
            candidate_entry = candidate_lookup.get(candidate_id)
            if not candidate_entry:
                continue
            candidate_player, candidate_logs = candidate_entry
            team = str(candidate_player.get("team") or "").upper()
            team_games = [
                game for game in get_team_regular_season_schedule(team, year)
                if (game.get("date") or "")[:10] < target_date.isoformat()
            ]
            team_games.sort(
                key=lambda game: game.get("date") or "", reverse=True
            )
            team_games = team_games[:10]

            # Garante que o próprio histórico do candidato seja a fonte
            # usada no split, sem uma segunda consulta de atleta.
            player_logs_by_id[candidate_id] = (candidate_player, candidate_logs)
            result["teammate_context"] = build_candidate_teammate_context(
                candidate_id=candidate_id,
                team=team,
                roster=roster_by_team.get(team, []),
                player_logs_by_id=player_logs_by_id,
                team_games=team_games,
                teammate_status_by_game=status_index
            )
    context_elapsed = time.perf_counter() - context_start
    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=contexto_companheiros candidatos=%s jogos_boxscore=%s tempo=%.2fs",
        selected_date,
        market,
        sum(1 for item in displayed_results if item.get("teammate_context")),
        len(league_baseline.get("player_status_by_game", {})) if league_baseline else 0,
        context_elapsed
    )

    return {
        "date": selected_date,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "count": len(displayed_results),
        "selection_summary": {
            "analyzed_candidates": len(results),
            "eligible_candidates": len(eligible_all_results),
            "displayed_candidates": len(displayed_results),
            "outside_top10_count": len(outside_top10_results),
            "unavailable_from_initial_top10_count": len(unavailable_top10_results),
            "unavailable_from_initial_top10": unavailable_top10_results,
            "outside_top10": outside_top10_results,
        },
        "source": "ESPN",
        "season": season_for(target_date),
        "teams_found": roster_data["teams_found"],
        "players_found": len(players),
        "players_consulted": successful_logs,
        "players_with_errors": errors,
        "availability": {
            "source": "ESPN",
            "report_status": availability_report_status,
            "warning": availability_report_warning,
            "reported_players": len(injury_index),
            "confirmed_out_removed": len(unavailable_top10_results),
            "uncertain_kept": sum(
                1 for item in displayed_results
                if (item.get("availability") or {}).get("availability") == "uncertain"
            ),
        },
        "updated_at": datetime.now().isoformat(),
        "audit_snapshots_saved": audit_snapshots_saved,
        "candidates": displayed_results
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NBA Props Scanner",
        "version": "3.17.0",
        "source": "ESPN"
    }


@app.get("/games")
def games(
    selected_date: Optional[str] = Query(
        default=None,
        alias="date"
    )
):

    if not selected_date:
        selected_date = date.today().isoformat()

    try:
        datetime.strptime(
            selected_date,
            "%Y-%m-%d"
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize AAAA-MM-DD."
        )

    # Mantém o retrospecto L5/L10 das equipes visível na agenda.
    # O histórico individual é aquecido em segundo plano enquanto o usuário
    # escolhe o mercado. Se a consulta começar antes do prewarm terminar,
    # /candidates espera o mesmo trabalho em vez de duplicar as 317 chamadas.
    schedule = fetch_espn_games(
        selected_date,
        enrich_context=True
    )

    teams = set()
    for game in schedule.get("games", []):
        teams.add(game["home"]["abbr"])
        teams.add(game["away"]["abbr"])

    if teams:
        schedule_player_history_prewarm(
            selected_date,
            teams
        )

    return schedule


@app.get("/candidates")
def candidates(
    selected_date: str = Query(
        alias="date"
    ),
    market: str = Query(
        default="points"
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=20
    )
):

    if market not in ACTIVE_MARKETS:
        raise HTTPException(
            status_code=400,
            detail="Mercado inválido ou inativo no Scanner."
        )

    try:
        datetime.strptime(
            selected_date,
            "%Y-%m-%d"
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize AAAA-MM-DD."
        )

    return collect_candidates(
        selected_date,
        market,
        limit
    )


@app.get("/capture-audit-snapshot", dependencies=[Depends(require_admin_api_key)])
def capture_audit_snapshot(
    selected_date: str = Query(alias="date"),
    market: str = Query(...)
):
    """Gera a população de auditoria sem registrar uso operacional do Scanner."""
    if market not in ACTIVE_MARKETS:
        raise HTTPException(
            status_code=400,
            detail="Mercado inválido ou inativo na auditoria."
        )
    if not DATABASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados não configurado."
        )

    try:
        datetime.strptime(selected_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize AAAA-MM-DD."
        )

    result = collect_candidates(
        selected_date,
        market,
        10,
        persist_operational=False
    )

    return {
        "status": "captured",
        "date": selected_date,
        "market": market,
        "eligible_players": result["selection_summary"]["eligible_candidates"],
        "snapshots_saved": result["audit_snapshots_saved"],
        "players_consulted": result["players_consulted"],
        "players_with_errors": result["players_with_errors"],
        "availability_report_status": result["availability"]["report_status"],
        "updated_at": result["updated_at"]
    }


@app.get("/refresh-audit", dependencies=[Depends(require_admin_api_key)])
def refresh_audit(
    selected_date: Optional[str] = Query(default=None, alias="date"),
    history: bool = Query(default=False),
):
    """
    Rotina híbrida da Auditoria.

    Modo diário:
    1) garante a captura do dia informado para todos os mercados ativos;
    2) tenta apurar pendências de resultados de datas anteriores à data informada.

    Modo histórico acumulado (?history=true):
    1) usa automaticamente a data atual de São Paulo;
    2) garante a captura de hoje para todos os mercados ativos;
    3) tenta apurar todas as combinações históricas pendentes, incluindo hoje.
    """
    if not DATABASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados não configurado."
        )

    if history:
        target_date = audit_history_cutoff_date()
        selected_date = target_date.isoformat()
    else:
        if not selected_date:
            raise HTTPException(
                status_code=400,
                detail="Data obrigatória para atualização diária."
            )
        try:
            target_date = datetime.strptime(
                selected_date,
                "%Y-%m-%d"
            ).date()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Data inválida. Utilize AAAA-MM-DD."
            )

    capture_results = []
    capture_failures = []

    # Não reconstruímos uma fotografia que já existe. A Auditoria é baseada
    # em snapshots imutáveis; recapturar uma data/mercado já capturado só
    # desperdiça CPU/memória e pode provocar picos no serviço.
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT market
                FROM audit_snapshot_runs
                WHERE selected_date = %s
                """,
                (target_date,)
            )
            captured_markets = {
                str(row[0]).strip()
                for row in cursor.fetchall()
            }

    for market in ACTIVE_MARKETS:
        if market in captured_markets:
            capture_results.append({
                "market": market,
                "status": "already_captured",
                "snapshots_saved": 0,
                "eligible_players": None,
                "players_consulted": 0,
                "players_with_errors": 0,
            })
            continue

        try:
            result = collect_candidates(
                selected_date,
                market,
                10,
                persist_operational=False
            )
            capture_results.append({
                "market": market,
                "status": "captured",
                "snapshots_saved": result.get("audit_snapshots_saved", 0),
                "eligible_players": result.get("selection_summary", {}).get("eligible_candidates", 0),
                "players_consulted": result.get("players_consulted", 0),
                "players_with_errors": result.get("players_with_errors", 0),
            })
        except Exception as error:
            logger.exception(
                "Falha na captura híbrida da auditoria: data=%s mercado=%s",
                selected_date,