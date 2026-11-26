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
                        selected_date,
                        market,
                        player_id
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

# Limite global de chamadas simultâneas à ESPN. Permite sobrepor histórico
# individual e defesa sem multiplicar indefinidamente a pressão externa.
_ESPN_REQUEST_SEMAPHORE = threading.BoundedSemaphore(16)


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

        with _ESPN_REQUEST_SEMAPHORE:
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

    fga_allowed = parse_attempts(stats.get("fieldGoalsMade-fieldGoalsAttempted"))
    three_pa_allowed = parse_attempts(stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))

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
            total_stat / total_minutes,
            4
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


@lru_cache(maxsize=8)
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
        max_workers=3
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
        "fgm"
    ]

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
            collect_roster_players(
                value,
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

    # A coleta de candidatos precisa apenas de confrontos e elencos.
    # O retrospecto L5 das equipes é carregado separadamente por /games,
    # evitando consultas extras no caminho crítico do Scanner.
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


@lru_cache(maxsize=200)
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
            or entry.get("id")
            or ""
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




def _extract_team_players_from_summary(summary, team_abbr):
    """Extrai todos os jogadores de uma equipe de um boxscore já carregado."""
    result = {}
    expected_team = str(team_abbr or "").upper()

    players_blocks = (
        summary.get("boxscore", {})
        .get("players", [])
    )
    if not isinstance(players_blocks, list):
        return result

    for team_block in players_blocks:
        if not isinstance(team_block, dict):
            continue

        block_team = (
            team_block.get("team", {})
            .get("abbreviation")
            or ""
        ).upper()

        if block_team != expected_team:
            continue

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

                athlete_id = str(athlete.get("id", "")).strip()
                if not athlete_id:
                    continue

                raw_stats = (
                    athlete_entry.get("stats")
                    or athlete_entry.get("statistics")
                )

                if raw_stats is None:
                    continue

                minutes = get_player_stat_value(
                    raw_stats, key_indexes, "minutes"
                )
                assists = get_player_stat_value(
                    raw_stats, key_indexes, "assists"
                )
                rebounds = get_player_stat_value(
                    raw_stats, key_indexes, "rebounds"
                )
                points = get_player_stat_value(
                    raw_stats, key_indexes, "points"
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

                result[athlete_id] = {
                    "player_id": athlete_id,
                    "player": (
                        athlete.get("displayName")
                        or athlete.get("fullName")
                        or athlete.get("shortName")
                        or ""
                    ),
                    "team": expected_team,
                    "minutes": minutes,
                    "assists": assists,
                    "rebounds": rebounds,
                    "points": points,
                    "fga": fga,
                    "three_pa": three_pa,
                    "two_pa": two_pa,
                    "fgm": fgm
                }

    return result


def build_shared_boxscore_context(
    players,
    target_date,
    year,
    opponent_teams
):
    """
    Busca uma única fotografia de boxscores recentes e deriva dela tanto
    histórico individual quanto contexto defensivo.

    O Scanner não usa game logs e boxscores em paralelo: primeiro constrói
    uma base compartilhada de partidas e depois apenas transforma os dados.

    Importante para o Render Free: os summaries ESPN são processados conforme
    chegam. Não mantemos um dicionário com dezenas/centenas de respostas
    completas em memória enquanto também mantemos player_logs e defesa.
    """
    started = time.perf_counter()

    teams = {
        str(player.get("team") or "").upper()
        for player in players
        if player.get("team")
    }
    teams.update(
        str(team or "").upper()
        for team in opponent_teams
        if team
    )

    team_games = {}
    schedule_errors = 0

    def fetch_team_games(team):
        schedule = get_team_regular_season_schedule(team, year)
        games = [
            game
            for game in schedule
            if (game.get("date") or "")[:10] < target_date.isoformat()
        ]
        games.sort(
            key=lambda game: game.get("date") or "",
            reverse=True
        )
        return team, games[:10]

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(fetch_team_games, team)
            for team in sorted(teams)
        ]
        for future in as_completed(futures):
            try:
                team, games = future.result()
                team_games[team] = games
            except Exception as error:
                schedule_errors += 1
                logger.warning(
                    "Falha ao montar jogos compartilhados de %s: %s",
                    team,
                    error
                )

    # Mapeia cada boxscore para as equipes que realmente o utilizam.
    # Isso evita percorrer todas as equipes para cada summary recebido.
    game_ids = {}
    game_teams = {}
    for team, games in team_games.items():
        for game in games:
            game_id = str(game.get("game_id") or "")
            if not game_id:
                continue
            game_ids[game_id] = game
            game_teams.setdefault(game_id, set()).add(team)

    player_logs = {
        str(player.get("id")): []
        for player in players
        if player.get("id")
    }

    defense_teams = {
        str(team or "").upper()
        for team in opponent_teams
        if team
    }
    defense_results = {
        team: []
        for team in defense_teams
    }

    summary_errors = 0
    summaries_ok = 0
    summary_stage_start = time.perf_counter()

    def fetch_summary(game_id):
        return game_id, get_game_summary(game_id)

    # O limite permanece baixo para não reproduzir o pico de memória/requisições
    # do desenho anterior. Cada summary é transformado e liberado imediatamente.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(fetch_summary, game_id)
            for game_id in game_ids
        ]
        for future in as_completed(futures):
            try:
                game_id, summary = future.result()
                if not summary:
                    continue

                summaries_ok += 1

                # Deriva o histórico individual diretamente deste boxscore.
                # O resumo deixa de ser guardado em um dicionário global.
                for team in game_teams.get(str(game_id), ()):
                    team_players = _extract_team_players_from_summary(
                        summary,
                        team
                    )

                    for player_id, player_game in team_players.items():
                        player_logs.setdefault(player_id, []).append({
                            **player_game,
                            "event_id": str(game_id),
                            "date": (game_ids[str(game_id)].get("date") or "")[:10],
                            "opponent": game_ids[str(game_id)].get("opponent"),
                            "home_away": game_ids[str(game_id)].get("home_away")
                        })

                # Deriva a defesa somente para as equipes que realmente são
                # adversárias na consulta atual.
                for team in defense_teams.intersection(
                    game_teams.get(str(game_id), set())
                ):
                    stats = extract_team_defense_stats(summary, team)
                    if not stats:
                        continue

                    position_data = aggregate_position_game(summary, team)
                    defense_results.setdefault(team, []).append({
                        **game_ids[str(game_id)],
                        **stats,
                        "positions": position_data["positions"]
                    })

            except Exception as error:
                summary_errors += 1
                logger.warning(
                    "Falha ao processar boxscore compartilhado: %s",
                    error
                )

    for player_id in player_logs:
        player_logs[player_id].sort(
            key=lambda game: game.get("date") or "",
            reverse=True
        )

    # Se já existem pelo menos 10 partidas da equipe, um atleta com menos
    # de 10 aparições nos 10 boxscores pode ter jogos mais antigos. Nesses
    # poucos casos usamos o endpoint individual como fallback para preservar
    # exatamente a semântica histórica do L10.
    fallback_players = []
    for player in players:
        player_id = str(player.get("id") or "")
        team = str(player.get("team") or "").upper()
        logs = player_logs.get(player_id, [])

        if len(team_games.get(team, [])) < 10 or len(logs) >= 10:
            continue

        previous_games = [
            game
            for game in logs
            if (game.get("date") or "") < target_date.isoformat()
        ]
        if len(previous_games) < 3:
            continue

        last5 = previous_games[:5]
        minutes_values = [
            game.get("minutes")
            for game in last5
            if game.get("minutes") is not None
        ]
        if not minutes_values:
            continue

        minutes_avg = sum(minutes_values) / len(minutes_values)
        if minutes_avg < 24:
            continue

        fallback_players.append(player)

    fallback_errors = 0

    def fetch_fallback(player):
        return player, get_athlete_logs(player["id"], year)

    if fallback_players:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(fetch_fallback, player)
                for player in fallback_players
            ]
            for future in as_completed(futures):
                try:
                    player, logs = future.result()
                    player_logs[str(player["id"])] = list(logs)
                except Exception as error:
                    fallback_errors += 1
                    logger.warning(
                        "Fallback de histórico falhou para %s: %s",
                        player.get("player"),
                        error
                    )

    defense_cache = {}

    for team, results in defense_results.items():
        results.sort(
            key=lambda game: game.get("date") or "",
            reverse=True
        )

        def calculate_averages(sample):
            averages = {}
            for key in (
                "fga_allowed",
                "three_pa_allowed",
                "two_pa_allowed",
                "fgm_allowed",
                "rebounds_allowed",
                "assists_allowed"
            ):
                values = [
                    game[key]
                    for game in sample
                    if game.get(key) is not None
                ]
                averages[key] = (
                    round(sum(values) / len(values), 1)
                    if values else None
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

                for stat_key in (
                    "points",
                    "assists",
                    "rebounds",
                    "fga",
                    "three_pa",
                    "two_pa",
                    "fgm"
                ):
                    game_values = []
                    individual_total = 0.0
                    individual_appearances = 0

                    for game in position_games:
                        position_data = (
                            game.get("positions", {})
                            .get(position, {})
                        )
                        value = position_data.get(stat_key)
                        if value is not None:
                            game_values.append(value)
                            individual_total += value

                        individual_appearances += (
                            position_data
                            .get("stat_players_count", {})
                            .get(stat_key, 0)
                        )

                    stat_games_analyzed[stat_key] = len(game_values)
                    averages[stat_key] = (
                        round(sum(game_values) / len(game_values), 1)
                        if game_values else None
                    )
                    stat_player_appearances[stat_key] = individual_appearances
                    individual_averages[stat_key] = (
                        round(
                            individual_total / individual_appearances,
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
                    "games_analyzed": len(position_games),
                    "players_per_game_avg": (
                        round(
                            sum(players_per_game) / len(players_per_game),
                            1
                        )
                        if players_per_game else 0
                    ),
                    "stat_games_analyzed": stat_games_analyzed,
                    "averages": averages,
                    "individual_averages": individual_averages,
                    "stat_player_appearances": stat_player_appearances
                }

            return position_results

        history = {
            "team": team,
            "games_analyzed": len(results),
            "games": results,
            "averages": calculate_averages(results),
            "positions": calculate_position_averages(results)
        }

        for window in (5, 10):
            sample = results[:window]
            history[f"L{window}"] = {
                "team": team,
                "games_analyzed": len(sample),
                "games": sample,
                "averages": calculate_averages(sample),
                "positions": calculate_position_averages(sample)
            }

        defense_cache[team] = history

    elapsed = time.perf_counter() - started
    logger.info(
        "Shared boxscore context: data=%s equipes=%s jogos_unicos=%s summaries_ok=%s "
        "summary_errors=%s fallback_players=%s fallback_errors=%s tempo_summaries=%.2fs total=%.2fs",
        target_date.isoformat(),
        len(teams),
        len(game_ids),
        summaries_ok,
        summary_errors,
        len(fallback_players),
        fallback_errors,
        time.perf_counter() - summary_stage_start,
        elapsed
    )

    return {
        "player_logs": player_logs,
        "defense_cache": defense_cache,
        "team_games": team_games,
        "games_unique": len(game_ids),
        "summaries_ok": summaries_ok,
        "summary_errors": summary_errors,
        "fallback_players": len(fallback_players),
        "fallback_errors": fallback_errors,
        "schedule_errors": schedule_errors,
        "elapsed": elapsed
    }


NBA_TEAM_ABBREVIATIONS = (
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"
)


@lru_cache(maxsize=2)
def get_league_position_defense_baseline(year, target_date_iso, include_player_status=False):
    """Média da liga por posição, processando boxscores sem reter a liga inteira em memória.

    O índice detalhado de status dos jogadores só é mantido quando uma rotina de
    auditoria realmente precisa dele. O Scanner operacional não precisa desse
    objeto pesado para calcular a defesa e o Score.
    """
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

    # Cada boxscore é processado assim que chega. Guardamos apenas as métricas
    # agregadas, amostras individuais de minutos e o índice compacto de status.
    # Antes, o dicionário summaries mantinha centenas de respostas ESPN completas
    # simultaneamente, além do cache global de boxscores.
    game_team_indexes = {}
    for team, games in team_samples.items():
        for index, game in enumerate(games[:10]):
            game_id = str(game.get("game_id") or "")
            if game_id:
                game_team_indexes.setdefault(game_id, []).append((team, index))

    game_ids = list(game_team_indexes.keys())
    player_status_by_game = {}
    processed_games = 0
    position_samples_by_team = {
        team: {}
        for team in team_samples
    }

    stat_keys = ("points", "assists", "rebounds", "fga", "three_pa", "two_pa", "fgm")
    samples = {
        window: {
            position: {stat: [] for stat in stat_keys}
            for position in ("G", "F", "C")
        }
        for window in (5, 10)
    }
    minute_player_samples = {
        window: {
            position: []
            for position in ("G", "F", "C")
        }
        for window in (5, 10)
    }

    def process_summary(game_id, summary):
        nonlocal processed_games
        if not summary:
            return

        processed_games += 1
        player_blocks = (summary.get("boxscore") or {}).get("players") or []
        team_abbreviations = {
            str((block.get("team") or {}).get("abbreviation") or "").upper()
            for block in player_blocks
            if (block.get("team") or {}).get("abbreviation")
        }
        if include_player_status:
            status_index = {}
            for team_abbr in team_abbreviations:
                status_index.update(extract_team_player_status(summary, team_abbr))
            player_status_by_game[str(game_id)] = status_index

        for team, index in game_team_indexes.get(str(game_id), []):
            try:
                positions = aggregate_position_game(summary, team).get("positions", {})
                position_samples_by_team[team][index] = positions
            except Exception as error:
                logger.warning("Falha ao agregar benchmark %s/%s: %s", team, game_id, error)

    def fetch_summary(game_id):
        return game_id, get_game_summary(game_id)

    # A defesa dos adversários usa até 12 consultas simultâneas. O benchmark
    # fica limitado a 4, evitando o pico combinado de até 24 boxscores.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_summary, game_id) for game_id in game_ids]
        for future in as_completed(futures):
            try:
                game_id, summary = future.result()
                process_summary(game_id, summary)
            except Exception as error:
                logger.warning("Falha ao consultar jogo para benchmark da liga: %s", error)


    # Reaplica a ordem original por equipe/data para manter as mesmas amostras
    # e a mesma ordem numérica do cálculo anterior, independentemente da rede.
    for team, games in team_samples.items():
        team_positions = position_samples_by_team.get(team, {})
        for index, _game in enumerate(games[:10]):
            positions = team_positions.pop(index, {})
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
        "unique_games": processed_games
    }
    if include_player_status:
        result["player_status_by_game"] = player_status_by_game
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
        target_date_iso, len(team_samples), processed_games,
        result["positions"]["G"]["assists"]["l5_games"],
        result["positions"]["G"]["assists"]["l10_games"],
        time.perf_counter() - started
    )
    return result

# Serializa a construção do benchmark por processo. Isso evita que o
# prewarm e a primeira consulta de candidatos montem o mesmo benchmark em paralelo.
_LEAGUE_BASELINE_SINGLEFLIGHT_LOCK = threading.Lock()


def get_league_position_defense_baseline_singleflight(year, target_date_iso, include_player_status=False):
    with _LEAGUE_BASELINE_SINGLEFLIGHT_LOCK:
        return get_league_position_defense_baseline(
            year,
            target_date_iso,
            include_player_status=include_player_status
        )


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
        started = time.perf_counter()
        try:
            target_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
            baseline = get_league_position_defense_baseline_singleflight(
                season_year(target_date),
                selected_date
            )
            elapsed = time.perf_counter() - started
            with _DEFENSE_PREWARM_LOCK:
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



def build_scanner_league_baseline_from_defense_cache(defense_cache):
    """Constrói uma referência leve usando as equipes já consultadas pelo Scanner.

    Não faz novas chamadas à ESPN. A referência usa exatamente os mesmos
    boxscores defensivos já carregados para os adversários dos jogos da data,
    preservando a mesma estrutura consumida por build_player_result().
    """
    stat_keys = (
        "points",
        "assists",
        "rebounds",
        "fga",
        "three_pa",
        "two_pa",
        "fgm",
    )
    positions = ("G", "F", "C")

    samples = {
        window: {
            position: {stat: [] for stat in stat_keys}
            for position in positions
        }
        for window in (5, 10)
    }
    minute_player_samples = {
        window: {position: [] for position in positions}
        for window in (5, 10)
    }

    teams_analyzed = 0
    unique_games = set()

    for team, defense in defense_cache.items():
        if not defense:
            continue
        teams_analyzed += 1

        for window in (5, 10):
            games = (
                defense.get(f"L{window}", {})
                .get("games", [])
            )

            for game in games:
                game_id = str(game.get("game_id") or "")
                if game_id:
                    unique_games.add(game_id)

                for position in positions:
                    position_data = (
                        game.get("positions", {})
                        .get(position, {})
                    )
                    players = position_data.get("players", [])

                    if players:
                        minute_player_samples[window][position].append({
                            "positions": {
                                position: {
                                    "players": players
                                }
                            }
                        })

                    for stat_key in stat_keys:
                        value = position_data.get(stat_key)
                        if value is not None:
                            samples[window][position][stat_key].append(value)

    result = {
        "positions": {},
        "minute_player_samples": minute_player_samples,
        "teams_analyzed": teams_analyzed,
        "unique_games": len(unique_games),
        "scope": "scanner_opponents",
    }

    for position in positions:
        result["positions"][position] = {}
        for stat_key in stat_keys:
            result["positions"][position][stat_key] = {}
            for window in (5, 10):
                values = samples[window][position][stat_key]
                result["positions"][position][stat_key][f"l{window}_avg"] = (
                    round(sum(values) / len(values), 2)
                    if values else None
                )
                result["positions"][position][stat_key][f"l{window}_games"] = len(values)

    logger.info(
        "Scanner league reference: equipes=%s jogos_unicos=%s escopo=adversarios_da_data",
        teams_analyzed,
        len(unique_games),
    )
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

        if league_baseline:
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

    recent_games = [
        {
            "date": game.get("date"),
            "opponent": game.get("opponent"),
            "home_away": game.get("home_away"),
            "minutes": game.get("minutes"),
            "value": game.get(stat_key)
        }
        for game in last10
    ]

    return {
        "player_id": str(player["id"]),
        "player": player["player"],
        "recent_games": recent_games,
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
        "opponent_position_individual_defense_player_games_l5": position_player_games_l5,
        "opponent_position_individual_defense_player_games_l10": position_player_games_l10,
        "opponent_position_minutes_filtered_threshold": round(minutes_avg, 1),
        "opponent_position_minute_band_reference_version": 1,
        "opponent_position_minute_band_margin": 5.0,
        "opponent_position_minute_band_center": round(minutes_avg, 1),
        "opponent_position_minute_band_lower": (
            position_minute_band_l5.get("minutes_lower")
            if position_minute_band_l5 else None
        ),
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

    # A aquisição do Scanner usa uma única camada de boxscores compartilhados.
    # Dela derivamos histórico individual e contexto defensivo, evitando que
    # centenas de endpoints de atleta e dezenas de boxscores concorram entre si.
    shared_context = build_shared_boxscore_context(
        players,
        target_date,
        year,
        set(opponents.values())
    )

    player_logs = shared_context["player_logs"]
    successful_logs = len(player_logs)
    errors = shared_context["fallback_errors"] + shared_context["summary_errors"]

    eligible_players = []
    for player in players:
        logs = player_logs.get(str(player["id"]), [])
        previous_games = [
            game
            for game in logs
            if game["date"] < target_date.isoformat()
        ]
        if len(previous_games) < 3:
            continue
        last5 = previous_games[:5]
        minutes_values = [
            game["minutes"]
            for game in last5
            if game.get("minutes") is not None
        ]
        if not minutes_values:
            continue
        minutes_avg = sum(minutes_values) / len(minutes_values)
        if minutes_avg < 24:
            continue
        eligible_players.append((player, tuple(logs)))

    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=historico_compartilhado "
        "jogadores=%s elegiveis=%s jogos_unicos=%s summaries=%s tempo=%.2fs",
        selected_date,
        market,
        successful_logs,
        len(eligible_players),
        shared_context["games_unique"],
        shared_context["summaries_ok"],
        shared_context["elapsed"]
    )

    eligible_opponents = {
        opponents.get(player["team"])
        for player, _ in eligible_players
        if opponents.get(player["team"])
    }

    defense_cache = {
        team: shared_context["defense_cache"].get(team, [])
        for team in eligible_opponents
    }

    defense_elapsed = shared_context["elapsed"]

    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=defesa_compartilhada "
        "adversarios=%s jogos_unicos=%s summaries=%s tempo=%.2fs",
        selected_date,
        market,
        len(eligible_opponents),
        shared_context["games_unique"],
        shared_context["summaries_ok"],
        defense_elapsed
    )

    # Mantém o benchmark leve do Scanner usando somente as equipes já consultadas.
    league_baseline = build_scanner_league_baseline_from_defense_cache(defense_cache)

    logger.info(
        "Candidatos cache: data=%s mercado=%s cache=get_team_defense_history "
        "shared_games=%s summaries=%s",
        selected_date, market,
        shared_context["games_unique"],
        shared_context["summaries_ok"]
    )
    logger.info(
        "Candidatos timing: data=%s mercado=%s etapa=defesa adversarios=%s tempo=%.2fs",
        selected_date, market, len(eligible_opponents), defense_elapsed
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

    # População completa para auditoria, sem limite de exibição.
    # Mantém-se separada de candidate_snapshots, que registra o uso operacional.
    # A captura completa da Auditoria roda somente em sua rotina própria
    # (/capture-audit-snapshot e /refresh-audit). Uma consulta do Scanner não
    # deve gravar novamente toda a população da Auditoria.
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

    # A agenda exibe o retrospecto recente das equipes.
    # O trabalho de contexto dos times fica fora do fluxo de coleta do Scanner.
    return fetch_espn_games(
        selected_date,
        enrich_context=True
    )


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
                market
            )
            capture_failures.append({
                "market": market,
                "error": str(error)
            })

    # A captura do dia informado é separada da apuração histórica.
    # Depois de garantir os snapshots do dia, procura combinações que já
    # possuem snapshots pendentes em datas anteriores e tenta atualizá-las.
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT s.selected_date, s.market
                FROM audit_snapshots s
                LEFT JOIN audit_results r
                    ON r.selected_date = s.selected_date
                    AND r.market = s.market
                    AND r.player_id = s.player_id
                LEFT JOIN audit_no_results nr
                    ON nr.selected_date = s.selected_date
                    AND nr.market = s.market
                    AND nr.player_id = s.player_id
                WHERE s.selected_date >= %s
                  AND s.selected_date <= %s
                  AND r.id IS NULL
                  AND nr.id IS NULL
                ORDER BY s.selected_date ASC, s.market ASC
                """,
                (
                    audit_history_start_date(),
                    target_date if history else target_date - timedelta(days=1)
                )
            )
            pending_combinations = cursor.fetchall()

    collection_results = []
    collection_failures = []

    for pending_date, market in pending_combinations:
        try:
            result = collect_results(
                selected_date=pending_date.isoformat(),
                market=market
            )
            collection_results.append({
                "date": pending_date.isoformat(),
                "market": market,
                "status": result.get("status"),
                "results_saved": result.get("audit_results_saved", 0),
                "dnp": result.get("audit_no_results", 0),
                "pending": result.get("audit_pending", 0),
            })
        except Exception as error:
            logger.exception(
                "Falha na apuração híbrida da auditoria: data=%s mercado=%s",
                pending_date,
                market
            )
            collection_failures.append({
                "date": pending_date.isoformat(),
                "market": market,
                "error": str(error)
            })

    return {
        "status": (
            "ok"
            if not capture_failures and not collection_failures
            else "partial"
        ),
        "date": selected_date,
        "history": history,
        "capture": {
            "markets": capture_results,
            "failures": capture_failures,
        },
        "collection": {
            "combinations_found": len(pending_combinations),
            "processed": collection_results,
            "failures": collection_failures,
        }
    }


@app.get("/collect-results", dependencies=[Depends(require_admin_api_key)])
def collect_results(
    selected_date: str = Query(
        alias="date"
    ),
    market: str = Query(
        default="assists"
    )
):

    if market not in MARKETS:
        raise HTTPException(
            status_code=400,
            detail="Mercado inválido."
        )

    actual_stat_keys = {
        "points": "points",
        "assists": "assists",
        "rebounds": "rebounds",
        "threes": "three_pa",
        "two_pa": "two_pa",
        "fgm": "fgm",
        "attempts": "fga"
    }

    actual_stat_key = actual_stat_keys[market]

    try:
        target_date = datetime.strptime(
            selected_date,
            "%Y-%m-%d"
        ).date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida."
        )

    if not DATABASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados não configurado."
        )

    schedule = fetch_espn_games(
        selected_date,
        enrich_context=False
    )

    completed_games = {}
    completed_game_ids = []
    completed_game_context = {}

    for game in schedule.get("games", []):

        if not game.get("completed"):
            continue

        home = game.get("home", {}).get("abbr")
        away = game.get("away", {}).get("abbr")
        game_id = str(game.get("game_id") or "")

        if home and away and game_id:
            completed_games[(home.upper(), away.upper())] = game_id
            completed_games[(away.upper(), home.upper())] = game_id
            completed_game_ids.append(game_id)
            completed_game_context[game_id] = {
                "matchup": f"{away.upper()}@{home.upper()}",
                "start_time_brt": game.get("start_time_brt"),
                "start_date_et": game.get("start_date_et"),
                "start_time_et": game.get("start_time_et"),
            }

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    'operational' AS snapshot_source,
                    player_id,
                    player_name,
                    team,
                    opponent,
                    payload
                FROM candidate_snapshots s
                WHERE s.selected_date = %s AND s.market = %s
                    AND NOT EXISTS (
                        SELECT 1
                        FROM candidate_results r
                        WHERE r.selected_date = s.selected_date
                            AND r.market = s.market
                            AND r.player_id = s.player_id
                    )

                UNION ALL

                SELECT
                    'audit' AS snapshot_source,
                    s.player_id,
                    s.player_name,
                    s.team,
                    s.opponent,
                    s.payload
                FROM audit_snapshots s
                WHERE s.selected_date = %s AND s.market = %s
                    AND NOT EXISTS (
                        SELECT 1
                        FROM audit_results r
                        WHERE r.selected_date = s.selected_date
                            AND r.market = s.market
                            AND r.player_id = s.player_id
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM audit_no_results nr
                        WHERE nr.selected_date = s.selected_date
                            AND nr.market = s.market
                            AND nr.player_id = s.player_id
                    )

                ORDER BY player_name, snapshot_source
                """,
                (target_date, market, target_date, market)
            )
            snapshots = cursor.fetchall()

    if not snapshots:
        return {
            "date": selected_date,
            "market": market,
            "status": "no_pending_snapshots",
            "results_saved_this_run": 0,
            "operational_results_saved": 0,
            "audit_results_saved": 0,
            "operational_pending": 0,
            "audit_pending": 0,
            "pending": 0
        }

    # Só consulta os boxscores dos jogos que realmente aparecem nos snapshots
    # pendentes desta data/mercado. Isso reduz chamadas desnecessárias e a
    # pressão de memória/CPU no serviço de staging.
    pending_game_ids = {
        completed_games.get(
            ((snapshot[3] or "").upper(), (snapshot[4] or "").upper())
        )
        for snapshot in snapshots
    }
    pending_game_ids.discard(None)

    summary_cache = {}

    def fetch_summary(game_id):
        return (
            game_id,
            get_game_summary(game_id)
        )

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        futures = [
            executor.submit(
                fetch_summary,
                game_id
            )
            for game_id in pending_game_ids
        ]

        for future in as_completed(futures):
            try:
                game_id, summary = future.result()
                summary_cache[game_id] = summary
            except Exception:
                pass

    operational_results_to_save = []
    audit_results_to_save = []
    audit_no_results_to_save = []
    audit_pending_reasons_to_save = []
    operational_pending = 0
    audit_pending = 0
    audit_no_result_count = 0

    for snapshot in snapshots:

        (
            snapshot_source,
            player_id,
            player_name,
            team,
            opponent,
            payload
        ) = snapshot

        target_results = (
            audit_results_to_save
            if snapshot_source == "audit"
            else operational_results_to_save
        )

        game_id = completed_games.get(
            (
                (team or "").upper(),
                (opponent or "").upper()
            )
        )

        if not game_id:
            if snapshot_source == "audit":
                audit_pending += 1
                audit_pending_reason = (
                    target_date, market, str(player_id),
                    "game_not_found",
                    "Confronto não localizado entre os jogos concluídos",
                    f"{team or '?'} x {opponent or '?'}"
                )
                audit_pending_reasons_to_save.append(audit_pending_reason)
            else:
                operational_pending += 1
            continue

        summary = summary_cache.get(game_id)

        if not summary:
            if snapshot_source == "audit":
                audit_pending += 1
                audit_pending_reasons_to_save.append((
                    target_date, market, str(player_id),
                    "summary_unavailable",
                    "Resumo da partida não disponível",
                    f"game_id={game_id}"
                ))
            else:
                operational_pending += 1
            continue

        player_game = (
            extract_player_game_from_summary(
                summary,
                player_id,
                player_name=player_name,
                team_abbr=team
            )
        )

        # Alguns boxscores da ESPN não trazem atletas inativos/DNP no bloco
        # tradicional de estatísticas. Antes de manter o caso como pendente,
        # consulta o roster da própria partida para confirmar didNotPlay.
        if not player_game and snapshot_source == "audit":
            try:
                roster_status = get_game_team_player_status(
                    game_id,
                    team,
                    player_id
                )
            except Exception as error:
                logger.warning(
                    "Falha ao consultar roster da partida para DNP: game_id=%s player_id=%s error=%s",
                    game_id,
                    player_id,
                    error
                )
                roster_status = None

            if roster_status and roster_status.get("did_not_play"):
                audit_no_results_to_save.append((
                    target_date,
                    market,
                    str(player_id),
                    player_name,
                    team,
                    opponent,
                    "did_not_play",
                    game_id,
                    Jsonb({
                        "minutes": 0.0,
                        "reason": roster_status.get("reason") or "did_not_play",
                        "game_date": selected_date,
                        "source": "game_roster"
                    })
                ))
                audit_no_result_count += 1
                continue

        # Segunda fonte independente: relatório oficial de lesões da NBA.
        # Só status explícito "Out" no relatório do confronto pode encerrar
        # como DNP; ausência do atleta ou falha da consulta mantém pendente.
        if not player_game and snapshot_source == "audit":
            game_context = completed_game_context.get(game_id, {})
            try:
                official_status = get_official_nba_injury_status(
                    selected_date=selected_date,
                    start_time_brt=game_context.get("start_time_brt"),
                    matchup=game_context.get("matchup"),
                    player_name=player_name,
                    start_date_et=game_context.get("start_date_et"),
                    start_time_et=game_context.get("start_time_et"),
                )
            except Exception as error:
                logger.warning(
                    "Falha ao consultar relatório oficial NBA: game_id=%s player=%s error=%s",
                    game_id, player_name, error
                )
                official_status = None

            if official_status and official_status.get("status") == "Out":
                audit_no_results_to_save.append((
                    target_date,
                    market,
                    str(player_id),
                    player_name,
                    team,
                    opponent,
                    "did_not_play",
                    game_id,
                    Jsonb({
                        "minutes": 0.0,
                        "reason": "official_nba_status_out",
                        "game_date": selected_date,
                        **official_status,
                    })
                ))
                audit_no_result_count += 1
                continue

        if not player_game:
            if snapshot_source == "audit":
                audit_pending += 1
                audit_pending_reasons_to_save.append((
                    target_date, market, str(player_id),
                    "player_not_found",
                    "Jogador não encontrado no boxscore da partida",
                    f"game_id={game_id}"
                ))
            else:
                operational_pending += 1
            continue

        minutes = player_game.get("minutes")

        if minutes is None or minutes <= 0:
            if snapshot_source == "audit":
                audit_no_results_to_save.append((
                    target_date,
                    market,
                    str(player_id),
                    player_name,
                    team,
                    opponent,
                    "did_not_play",
                    game_id,
                    Jsonb({
                        "minutes": minutes,
                        "reason": "did_not_play",
                        "game_date": selected_date
                    })
                ))
                audit_no_result_count += 1
            else:
                operational_pending += 1
            continue

        actual = player_game.get(actual_stat_key)

        if actual is None:
            if snapshot_source == "audit":
                audit_pending += 1
                audit_pending_reasons_to_save.append((
                    target_date, market, str(player_id),
                    "stat_unavailable",
                    "Estatística do mercado não disponível no boxscore",
                    f"game_id={game_id}; player_id={player_id}"
                ))
            else:
                operational_pending += 1
            continue

        target_results.append((
            target_date,
            market,
            str(player_id),
            player_name,
            team,
            opponent,
            float(actual),
            game_id,
            Jsonb({
                "minutes": minutes,
                actual_stat_key: actual,
                "game_date": selected_date
            })
        ))

    operational_inserted = 0
    audit_inserted = 0

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            for row in operational_results_to_save:
                cursor.execute(
                    """
                    INSERT INTO candidate_results (
                        selected_date, market, player_id, player_name,
                        team, opponent, actual_value, game_id, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (selected_date, market, player_id)
                    DO NOTHING
                    RETURNING id
                    """,
                    row
                )
                if cursor.fetchone():
                    operational_inserted += 1

            for row in audit_results_to_save:
                cursor.execute(
                    """
                    INSERT INTO audit_results (
                        selected_date, market, player_id, player_name,
                        team, opponent, actual_value, game_id, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (selected_date, market, player_id)
                    DO NOTHING
                    RETURNING id
                    """,
                    row
                )
                if cursor.fetchone():
                    audit_inserted += 1

            for row in audit_no_results_to_save:
                cursor.execute(
                    """
                    INSERT INTO audit_no_results (
                        selected_date, market, player_id, player_name,
                        team, opponent, reason, game_id, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (selected_date, market, player_id)
                    DO NOTHING
                    RETURNING id
                    """,
                    row
                )

            # Atualiza o motivo mais recente das pendências técnicas.
            for row in audit_pending_reasons_to_save:
                cursor.execute(
                    """
                    INSERT INTO audit_pending_reasons (
                        selected_date, market, player_id,
                        reason_code, reason, detail, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (selected_date, market, player_id)
                    DO UPDATE SET
                        reason_code = EXCLUDED.reason_code,
                        reason = EXCLUDED.reason,
                        detail = EXCLUDED.detail,
                        updated_at = NOW()
                    """,
                    row
                )

            # Um resultado concluído ou encerrado como DNP deixa de ter
            # motivo de pendência persistido.
            cursor.execute(
                """
                DELETE FROM audit_pending_reasons p
                WHERE p.selected_date = %s
                  AND p.market = %s
                  AND EXISTS (
                      SELECT 1
                      FROM audit_results r
                      WHERE r.selected_date = p.selected_date
                        AND r.market = p.market
                        AND r.player_id = p.player_id
                  )
                """,
                (target_date, market)
            )
            cursor.execute(
                """
                DELETE FROM audit_pending_reasons p
                WHERE p.selected_date = %s
                  AND p.market = %s
                  AND EXISTS (
                      SELECT 1
                      FROM audit_no_results nr
                      WHERE nr.selected_date = p.selected_date
                        AND nr.market = p.market
                        AND nr.player_id = p.player_id
                  )
                """,
                (target_date, market)
            )

    return {
        "date": selected_date,
        "market": market,
        "status": "processed",
        "results_saved_this_run": operational_inserted + audit_inserted,
        "operational_results_saved": operational_inserted,
        "audit_results_saved": audit_inserted,
        "operational_pending": operational_pending,
        "audit_pending": audit_pending,
        "audit_no_results": audit_no_result_count,
        "audit_pending_reasons_saved": len(audit_pending_reasons_to_save),
        "pending": operational_pending + audit_pending
    }


@app.get("/collect-pending-results", dependencies=[Depends(require_admin_api_key)])
def collect_pending_results(
    max_combinations: int = Query(default=100, ge=1, le=200)
):
    """
    Coleta resultados pendentes das populações operacional e de auditoria.

    Cada população é consultada e gravada em sua própria tabela de resultados.
    A combinação data/mercado é processada uma única vez, mesmo quando há
    pendências nas duas populações.
    """

    if not DATABASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados não configurado."
        )

    today = date.today()

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    pending.selected_date,
                    pending.market
                FROM (
                    SELECT s.selected_date, s.market
                    FROM candidate_snapshots s
                    LEFT JOIN candidate_results r
                        ON r.selected_date = s.selected_date
                        AND r.market = s.market
                        AND r.player_id = s.player_id
                    WHERE r.id IS NULL
                        AND s.selected_date <= %s
                        AND s.market = ANY(%s)

                    UNION

                    SELECT s.selected_date, s.market
                    FROM audit_snapshots s
                    LEFT JOIN audit_results r
                        ON r.selected_date = s.selected_date
                        AND r.market = s.market
                        AND r.player_id = s.player_id
                    LEFT JOIN audit_no_results nr
                        ON nr.selected_date = s.selected_date
                        AND nr.market = s.market
                        AND nr.player_id = s.player_id
                    WHERE r.id IS NULL
                        AND nr.id IS NULL
                        AND s.selected_date <= %s
                        AND s.market = ANY(%s)
                ) AS pending
                ORDER BY pending.selected_date ASC, pending.market ASC
                LIMIT %s
                """,
                (
                    today,
                    list(ACTIVE_MARKETS),
                    today,
                    list(ACTIVE_MARKETS),
                    max_combinations
                )
            )
            pending_combinations = cursor.fetchall()

    processed = []
    total_saved = 0
    total_pending = 0
    total_operational_saved = 0
    total_audit_saved = 0
    total_operational_pending = 0
    total_audit_pending = 0
    total_errors = 0

    for selected_date, market in pending_combinations:
        try:
            result = collect_results(
                selected_date=selected_date.isoformat(),
                market=market
            )

            saved = int(
                result.get("results_saved_this_run", 0)
            )
            pending = int(
                result.get("pending", 0)
            )

            total_saved += saved
            total_pending += pending
            total_operational_saved += int(result.get("operational_results_saved", 0))
            total_audit_saved += int(result.get("audit_results_saved", 0))
            total_operational_pending += int(result.get("operational_pending", 0))
            total_audit_pending += int(result.get("audit_pending", 0))

            processed.append({
                "date": selected_date.isoformat(),
                "market": market,
                "results_saved": saved,
                "operational_results_saved": result.get("operational_results_saved", 0),
                "audit_results_saved": result.get("audit_results_saved", 0),
                "operational_pending": result.get("operational_pending", 0),
                "audit_pending": result.get("audit_pending", 0),
                "pending": pending,
                "status": result.get("status", "processed")
            })

        except Exception as error:
            total_errors += 1

            logger.exception(
                "Falha na coleta automática: data=%s mercado=%s",
                selected_date,
                market
            )

            processed.append({
                "date": selected_date.isoformat(),
                "market": market,
                "results_saved": 0,
                "pending": None,
                "status": "error",
                "error": str(error)
            })

    return {
        "status": "processed",
        "combinations_found": len(pending_combinations),
        "combinations_limit": max_combinations,
        "results_saved_this_run": total_saved,
        "operational_results_saved": total_operational_saved,
        "audit_results_saved": total_audit_saved,
        "operational_pending": total_operational_pending,
        "audit_pending": total_audit_pending,
        "pending_after_processing": total_pending,
        "errors": total_errors,
        "processed": processed
    }


@app.get("/audit-history-coverage")
def audit_history_coverage(
    market: str = Query(...)
):
    """
    Consulta somente leitura da cobertura da auditoria independente.
    Retorna a quantidade de snapshots, resultados, DNPs e pendências
    por data, além do primeiro e último dia efetivamente encontrados.
    """
    if market not in ACTIVE_MARKETS:
        raise HTTPException(status_code=400, detail="Mercado inválido ou inativo na auditoria.")

    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH snapshot_counts AS (
                    SELECT selected_date, COUNT(*) AS snapshots_total
                    FROM audit_snapshots
                    WHERE market = %s
                    GROUP BY selected_date
                ),
                result_counts AS (
                    SELECT selected_date, COUNT(*) AS completed_results
                    FROM audit_results
                    WHERE market = %s
                    GROUP BY selected_date
                ),
                no_result_counts AS (
                    SELECT selected_date, COUNT(*) AS no_results
                    FROM audit_no_results
                    WHERE market = %s
                    GROUP BY selected_date
                )
                SELECT
                    s.selected_date,
                    s.snapshots_total,
                    COALESCE(r.completed_results, 0),
                    COALESCE(n.no_results, 0)
                FROM snapshot_counts s
                LEFT JOIN result_counts r ON r.selected_date = s.selected_date
                LEFT JOIN no_result_counts n ON n.selected_date = s.selected_date
                ORDER BY s.selected_date
                """,
                (market, market, market)
            )
            rows = cursor.fetchall()

    days = []
    for selected_date, snapshots_total, completed_results, no_results in rows:
        snapshots = int(snapshots_total or 0)
        completed = int(completed_results or 0)
        dnp = int(no_results or 0)
        days.append({
            "date": selected_date.isoformat(),
            "snapshots": snapshots,
            "completed_results": completed,
            "no_results": dnp,
            "pending_results": max(snapshots - completed - dnp, 0),
        })

    return {
        "status": "ok",
        "market": market,
        "period": {
            "first_date": days[0]["date"] if days else None,
            "last_date": days[-1]["date"] if days else None,
        },
        "days": days,
    }


@app.get("/audit-collection-status")
def audit_collection_status(
    date: str = Query(...)
):
    """
    Consulta somente leitura do estado já persistido da coleta.
    Não executa coleta nem altera snapshots/resultados.
    """
    try:
        selected_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Data inválida. Use YYYY-MM-DD.")

    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    'audit' AS source,
                    s.market,
                    COUNT(*) AS snapshots_total,
                    COUNT(r.player_id) AS completed_results,
                    COUNT(nr.player_id) AS no_results
                FROM audit_snapshots s
                LEFT JOIN audit_results r
                    ON r.selected_date = s.selected_date
                    AND r.market = s.market
                    AND r.player_id = s.player_id
                LEFT JOIN audit_no_results nr
                    ON nr.selected_date = s.selected_date
                    AND nr.market = s.market
                    AND nr.player_id = s.player_id
                WHERE s.selected_date = %s
                GROUP BY s.market

                UNION ALL

                SELECT
                    'operational' AS source,
                    s.market,
                    COUNT(*) AS snapshots_total,
                    COUNT(r.player_id) AS completed_results,
                    0 AS no_results
                FROM candidate_snapshots s
                LEFT JOIN candidate_results r
                    ON r.selected_date = s.selected_date
                    AND r.market = s.market
                    AND r.player_id = s.player_id
                WHERE s.selected_date = %s
                GROUP BY s.market

                ORDER BY market, source
                """,
                [selected_date, selected_date]
            )
            rows = cursor.fetchall()

    markets = {}
    for source, market, snapshots_total, completed_results, no_results in rows:
        total = int(snapshots_total or 0)
        completed = int(completed_results or 0)
        no_result_count = int(no_results or 0)
        markets.setdefault(market, {})[source] = {
            "snapshots_total": total,
            "completed_results": completed,
            "no_results": no_result_count,
            "pending_results": max(total - completed - no_result_count, 0),
        }

    # Motivos persistidos são apenas para pendências técnicas ainda abertas.
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT market, reason_code, reason, COUNT(*)
                FROM audit_pending_reasons
                WHERE selected_date = %s
                GROUP BY market, reason_code, reason
                ORDER BY market, reason_code
                """,
                (selected_date,)
            )
            reason_rows = cursor.fetchall()

    for market, reason_code, reason, count in reason_rows:
        markets.setdefault(market, {}).setdefault("audit", {}).setdefault("pending_reasons", []).append({
            "reason_code": reason_code,
            "reason": reason,
            "count": int(count or 0),
        })

    return {
        "status": "ok",
        "date": date,
        "markets": markets,
    }



@app.get("/audit-pending-details")
def audit_pending_details(
    date: str = Query(...),
    market: str = Query(...)
):
    """
    Lista somente as pendências ainda abertas de um mercado/data,
    com motivo curto e persistido. Não altera nenhum dado.
    """
    if market not in ACTIVE_MARKETS:
        raise HTTPException(status_code=400, detail="Mercado inválido.")

    try:
        selected_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Data inválida. Use YYYY-MM-DD.")

    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.player_id,
                    s.player_name,
                    s.team,
                    s.opponent,
                    COALESCE(
                        pr.reason,
                        'Motivo ainda não registrado'
                    ) AS reason,
                    pr.reason_code,
                    pr.detail,
                    s.captured_at
                FROM audit_snapshots s
                LEFT JOIN audit_results r
                    ON r.selected_date = s.selected_date
                    AND r.market = s.market
                    AND r.player_id = s.player_id
                LEFT JOIN audit_no_results nr
                    ON nr.selected_date = s.selected_date
                    AND nr.market = s.market
                    AND nr.player_id = s.player_id
                LEFT JOIN audit_pending_reasons pr
                    ON pr.selected_date = s.selected_date
                    AND pr.market = s.market
                    AND pr.player_id = s.player_id
                WHERE s.selected_date = %s
                  AND s.market = %s
                  AND r.player_id IS NULL
                  AND nr.player_id IS NULL
                ORDER BY s.audit_rank, s.player_name
                """,
                (selected_date, market)
            )
            rows = cursor.fetchall()

    return {
        "status": "ok",
        "date": date,
        "market": market,
        "pending": [
            {
                "player_id": str(player_id),
                "player_name": player_name,
                "team": team,
                "opponent": opponent,
                "reason": reason,
                "reason_code": reason_code,
                "detail": detail,
                "captured_at": captured_at.isoformat() if captured_at else None,
            }
            for (
                player_id,
                player_name,
                team,
                opponent,
                reason,
                reason_code,
                detail,
                captured_at,
            ) in rows
        ],
    }


@app.get("/audit-collection-diagnostics")
def audit_collection_diagnostics(
    date: str = Query(...)
):
    """
    Diagnóstico somente leitura das pendências da população independente
    de auditoria. Não altera snapshots nem resultados.
    """
    try:
        selected_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Data inválida. Use YYYY-MM-DD.")

    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    actual_stat_keys = {
        "points": "points",
        "assists": "assists",
        "rebounds": "rebounds",
        "threes": "three_pa",
        "two_pa": "two_pa",
        "fgm": "fgm",
        "attempts": "fga"
    }

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.market,
                    s.player_id,
                    s.player_name,
                    s.team,
                    s.opponent,
                    s.availability_status,
                    s.captured_at,
                    s.payload,
                    run.availability_report_status,
                    run.captured_at
                FROM audit_snapshots s
                LEFT JOIN audit_results r
                    ON r.selected_date = s.selected_date
                    AND r.market = s.market
                    AND r.player_id = s.player_id
                LEFT JOIN audit_no_results nr
                    ON nr.selected_date = s.selected_date
                    AND nr.market = s.market
                    AND nr.player_id = s.player_id
                LEFT JOIN audit_snapshot_runs run
                    ON run.selected_date = s.selected_date
                    AND run.market = s.market
                WHERE s.selected_date = %s
                    AND r.id IS NULL
                    AND nr.id IS NULL
                ORDER BY s.market, s.player_name
                """,
                (selected_date,)
            )
            pending_snapshots = cursor.fetchall()

    if not pending_snapshots:
        return {
            "status": "ok",
            "date": date,
            "pending_total": 0,
            "diagnostics": []
        }

    schedule = fetch_espn_games(date)
    completed_games = {}
    completed_game_ids = set()

    for game in schedule.get("games", []):
        if not game.get("completed"):
            continue
        home = game.get("home", {}).get("abbr")
        away = game.get("away", {}).get("abbr")
        game_id = str(game.get("game_id") or "")
        if home and away and game_id:
            completed_games[(home.upper(), away.upper())] = game_id
            completed_games[(away.upper(), home.upper())] = game_id
            completed_game_ids.add(game_id)

    summary_cache = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(get_game_summary, game_id): game_id
            for game_id in completed_game_ids
        }
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                summary_cache[game_id] = future.result()
            except Exception as error:
                logger.warning(
                    "Diagnóstico auditoria: falha no resumo game_id=%s: %s",
                    game_id,
                    error
                )
                summary_cache[game_id] = None

    diagnostics = []

    for (
        market, player_id, player_name, team, opponent,
        snapshot_availability_status, snapshot_captured_at, snapshot_payload,
        run_availability_report_status, run_captured_at
    ) in pending_snapshots:
        game_id = completed_games.get(
            ((team or "").upper(), (opponent or "").upper())
        )

        reason_code = "ok"
        reason = "Sem pendência identificada"
        detail = None

        if not game_id:
            reason_code = "game_not_found"
            reason = "Confronto não localizado entre os jogos concluídos"
            detail = f"{team or '?'} x {opponent or '?'}"
        else:
            summary = summary_cache.get(game_id)

            if not summary:
                reason_code = "summary_unavailable"
                reason = "Resumo da partida não disponível"
                detail = f"game_id={game_id}"
            else:
                diagnostic_player = find_player_in_summary_for_diagnostic(
                    summary,
                    player_id,
                    player_name
                )

                if not diagnostic_player:
                    reason_code = "player_not_found"
                    reason = "Jogador não encontrado no boxscore da partida"
                    detail = f"game_id={game_id}"
                elif diagnostic_player.get("match_method") == "name":
                    reason_code = "id_mismatch"
                    reason = "Jogador encontrado por nome, mas o ID do snapshot não coincide"
                    detail = (
                        f"game_id={game_id}; "
                        f"snapshot_player_id={player_id}; "
                        f"boxscore_player_id={diagnostic_player.get('boxscore_player_id')}"
                    )
                else:
                    minutes = diagnostic_player.get("boxscore_minutes")

                    if minutes is None or minutes <= 0:
                        reason_code = "zero_minutes"
                        reason = "Jogador não registrou minutos"
                        detail = (
                            f"game_id={game_id}; "
                            f"boxscore_player_id={diagnostic_player.get('boxscore_player_id')}; "
                            f"minutos={minutes}"
                        )
                    else:
                        reason_code = "unexpected"
                        reason = "Resultado encontrado, mas não foi persistido"
                        detail = (
                            f"game_id={game_id}; "
                            f"boxscore_player_id={diagnostic_player.get('boxscore_player_id')}; "
                            f"minutos={minutes}"
                        )

        diagnostics.append({
            "market": market,
            "player_id": str(player_id),
            "player_name": player_name,
            "team": team,
            "opponent": opponent,
            "game_id": game_id,
            "reason_code": reason_code,
            "reason": reason,
            "detail": detail,
            "boxscore_player_id": (
                diagnostic_player.get("boxscore_player_id")
                if game_id and summary and diagnostic_player
                else None
            ),
            "boxscore_player_name": (
                diagnostic_player.get("boxscore_player_name")
                if game_id and summary and diagnostic_player
                else None
            ),
            "boxscore_team": (
                diagnostic_player.get("boxscore_team")
                if game_id and summary and diagnostic_player
                else None
            ),
            "boxscore_minutes": (
                diagnostic_player.get("boxscore_minutes")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_statistics_keys": (
                diagnostic_player.get("boxscore_statistics_keys")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_stats_type": (
                diagnostic_player.get("boxscore_stats_type")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_stats_length": (
                diagnostic_player.get("boxscore_stats_length")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_min_index": (
                diagnostic_player.get("boxscore_min_index")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_raw_min_value": (
                diagnostic_player.get("boxscore_raw_min_value")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_athlete_entry_keys": (
                diagnostic_player.get("boxscore_athlete_entry_keys")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_athlete_keys": (
                diagnostic_player.get("boxscore_athlete_keys")
                if game_id and summary and diagnostic_player else None
            ),
            "boxscore_did_not_play": (
                diagnostic_player.get("boxscore_did_not_play")
                if game_id and summary and diagnostic_player else None
            ),
            "match_method": (
                diagnostic_player.get("match_method")
                if game_id and summary and diagnostic_player
                else None
            ),
            "snapshot_availability_status": snapshot_availability_status,
            "snapshot_availability_detail": (
                snapshot_payload.get("availability")
                if isinstance(snapshot_payload, dict)
                else None
            ),
            "snapshot_captured_at": (
                snapshot_captured_at.isoformat()
                if snapshot_captured_at else None
            ),
            "audit_run_availability_report_status": run_availability_report_status,
            "audit_run_captured_at": (
                run_captured_at.isoformat()
                if run_captured_at else None
            )
        })

    return {
        "status": "ok",
        "date": date,
        "pending_total": len(diagnostics),
        "diagnostics": diagnostics
    }


@app.get("/validation-history-summary")
def validation_history_summary(
    market: str = Query(default="assists"),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    cohort: str = Query(default="all")
):
    """Resumo leve da auditoria para a tabela comparativa por mercado.

    Mantém a mesma população e as mesmas definições da /validation-history,
    mas calcula apenas os KPIs necessários para a comparação dos seis mercados.
    """
    if market not in ACTIVE_MARKETS:
        raise HTTPException(status_code=400, detail="Mercado inválido ou inativo na auditoria.")
    if cohort not in {"top10", "all"}:
        raise HTTPException(status_code=400, detail="População de auditoria inválida.")

    try:
        parsed_from = datetime.strptime(date_from, "%Y-%m-%d").date() if date_from else None
        parsed_to = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Período de datas inválido.")
    # Sem datas: histórico acumulado da coorte validada, desde 19/11.
    if not date_from and not date_to:
        parsed_from = audit_metrics_start_date()
        parsed_to = audit_history_cutoff_date()


    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    query = """
        WITH snapshot_source AS (
            SELECT
                selected_date, market, player_id, triage_score,
                last5_avg, last10_avg, audit_rank AS display_rank,
                'audit' AS snapshot_source
            FROM audit_snapshots
            WHERE market = %s

        ),
        filtered AS (
            SELECT *,
                COUNT(*) FILTER (WHERE display_rank IS NULL)
                    OVER (PARTITION BY selected_date, market) AS missing_rank_count
            FROM snapshot_source
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY selected_date, market
                    ORDER BY
                        CASE WHEN missing_rank_count = 0 THEN display_rank END ASC NULLS LAST,
                        triage_score DESC NULLS LAST,
                        player_id ASC
                ) AS cohort_rank
            FROM filtered
        ),
        resolved AS (
            SELECT
                r.selected_date,
                r.player_id,
                r.triage_score,
                r.last5_avg,
                r.last10_avg,
                ar.actual_value AS actual_value,
                CASE
                    WHEN ar.id IS NOT NULL THEN 'completed'
                    WHEN nr.id IS NOT NULL THEN 'no_result'
                    ELSE 'pending'
                END AS result_status
            FROM ranked r
            LEFT JOIN audit_results ar
              ON r.snapshot_source = 'audit'
             AND ar.selected_date = r.selected_date
             AND ar.market = r.market
             AND ar.player_id = r.player_id
            LEFT JOIN audit_no_results nr
              ON r.snapshot_source = 'audit'
             AND nr.selected_date = r.selected_date
             AND nr.market = r.market
             AND nr.player_id = r.player_id
            WHERE (%s = 'all' OR r.cohort_rank <= 10)
    """

    params = [market, cohort]

    if parsed_from:
        query += " AND r.selected_date >= %s"
        params.append(parsed_from)
    if parsed_to:
        query += " AND r.selected_date <= %s"
        params.append(parsed_to)

    query += """
        )
        SELECT
            COUNT(*) AS snapshots_total,
            COUNT(*) FILTER (WHERE result_status = 'completed') AS completed_results,
            COUNT(*) FILTER (WHERE result_status = 'no_result') AS no_result_results,
            COUNT(*) FILTER (WHERE result_status = 'pending') AS pending_results,
            AVG(actual_value) FILTER (WHERE result_status = 'completed') AS actual_average,
            AVG(triage_score) FILTER (WHERE result_status = 'completed' AND triage_score IS NOT NULL) AS score_average,
            AVG(ABS(actual_value - last5_avg)) FILTER (
                WHERE result_status = 'completed' AND actual_value IS NOT NULL AND last5_avg IS NOT NULL
            ) AS l5_mae,
            AVG(ABS(actual_value - last10_avg)) FILTER (
                WHERE result_status = 'completed' AND actual_value IS NOT NULL AND last10_avg IS NOT NULL
            ) AS l10_mae,
            AVG(last5_avg) FILTER (
                WHERE result_status = 'completed' AND last5_avg IS NOT NULL
            ) AS l5_prediction_average,
            AVG(last10_avg) FILTER (
                WHERE result_status = 'completed' AND last10_avg IS NOT NULL
            ) AS l10_prediction_average,
            CORR(triage_score, actual_value) FILTER (
                WHERE result_status = 'completed' AND triage_score IS NOT NULL AND actual_value IS NOT NULL
            ) AS score_actual_pearson,
            MIN(selected_date) AS first_date,
            MAX(selected_date) AS last_date
        FROM resolved
    """

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

    snapshots_total = int(row[0] or 0)
    completed_results = int(row[1] or 0)
    no_result_results = int(row[2] or 0)
    pending_results = int(row[3] or 0)
    l5_mae = float(row[6]) if row[6] is not None else None
    l10_mae = float(row[7]) if row[7] is not None else None
    l5_prediction_average = float(row[8]) if row[8] is not None else None
    l10_prediction_average = float(row[9]) if row[9] is not None else None

    return {
        "status": "ok",
        "market": market,
        "market_label": MARKETS[market]["label"],
        "cohort": cohort,
        "period": {
            "first_date": row[11].isoformat() if row[11] is not None else None,
            "last_date": row[12].isoformat() if row[12] is not None else None
        },
        "summary": {
            "snapshots_total": snapshots_total,
            "completed_results": completed_results,
            "no_result_results": no_result_results,
            "pending_results": pending_results,
            "completion_rate_pct": round(
                (completed_results + no_result_results) / snapshots_total * 100, 2
            ) if snapshots_total else 0,
            "actual_average": round(float(row[4]), 2) if row[4] is not None else None,
            "score_average": round(float(row[5]), 2) if row[5] is not None else None,
            "l5_mae": round(l5_mae, 2) if l5_mae is not None else None,
            "l10_mae": round(l10_mae, 2) if l10_mae is not None else None,
            "l5_relative_error_pct": round(l5_mae / l5_prediction_average * 100, 2)
                if l5_mae is not None and l5_prediction_average not in (None, 0) else None,
            "l10_relative_error_pct": round(l10_mae / l10_prediction_average * 100, 2)
                if l10_mae is not None and l10_prediction_average not in (None, 0) else None,
            "score_actual_pearson": round(float(row[10]), 4) if row[10] is not None else None
        }
    }


@app.get("/validation-history")
def validation_history(
    market: str = Query(
        default="assists"
    ),
    date_from: Optional[str] = Query(
        default=None
    ),
    date_to: Optional[str] = Query(
        default=None
    ),
    cohort: str = Query(default="all")
):
    """
    Validação histórica aprimorada para exibir:
    - desempenho acumulado por faixa de score;
    - comparação entre L5 e L10;
    - erro médio das referências (MAE);
    - evolução dos resultados por dia;
    - desempenho observado da produção real por faixa de Score
      (sem interpretação como calibração probabilística).
    """

    if market not in ACTIVE_MARKETS:
        raise HTTPException(
            status_code=400,
            detail="Mercado inválido ou inativo na auditoria."
        )
    if cohort not in {"top10", "all"}:
        raise HTTPException(status_code=400, detail="População de auditoria inválida.")

    parsed_from = None
    parsed_to = None

    if date_from:
        parsed_from = datetime.strptime(
            date_from,
            "%Y-%m-%d"
        ).date()

    if date_to:
        parsed_to = datetime.strptime(
            date_to,
            "%Y-%m-%d"
        ).date()
    # Histórico acumulado da coorte validada: 19/11 até hoje.
    if not date_from and not date_to:
        parsed_from = audit_metrics_start_date()
        parsed_to = audit_history_cutoff_date()


    if not DATABASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados não configurado."
        )

    with get_database_connection() as connection:
        with connection.cursor() as cursor:

            query = """
                WITH snapshot_source AS (
                    SELECT
                        selected_date, market, player_id, player_name,
                        team, opponent, position, triage_score, last5_avg,
                        last10_avg, minutes_last5_avg, payload,
                        audit_rank AS display_rank,
                        'audit' AS snapshot_source
                    FROM audit_snapshots
                    WHERE market = %s

                        ),
                snapshot_cohort AS (
                    SELECT
                        s.*,
                        COUNT(*) FILTER (
                            WHERE s.display_rank IS NULL
                        ) OVER (
                            PARTITION BY s.selected_date, s.market
                        ) AS missing_rank_count
                    FROM snapshot_source s
                    WHERE 1 = 1
            """

            params = [market]

            if parsed_from:
                query += " AND s.selected_date >= %s"
                params.append(parsed_from)

            if parsed_to:
                query += " AND s.selected_date <= %s"
                params.append(parsed_to)

            query += """
                ),
                ranked_snapshots AS (
                    SELECT
                        selected_date, market, player_id, player_name,
                        team, opponent, position, triage_score, last5_avg,
                        last10_avg, minutes_last5_avg, payload, snapshot_source,
                        ROW_NUMBER() OVER (
                            PARTITION BY selected_date, market
                            ORDER BY
                                CASE WHEN missing_rank_count = 0
                                     THEN display_rank END ASC NULLS LAST,
                                triage_score DESC NULLS LAST,
                                player_id ASC
                        ) AS cohort_rank
                    FROM snapshot_cohort
                )
                SELECT
                    s.selected_date,
                    s.player_id,
                    s.player_name,
                    s.team,
                    s.opponent,
                    s.position,
                    s.triage_score,
                    s.last5_avg,
                    s.last10_avg,
                    s.minutes_last5_avg,
                    ar.actual_value,
                    CASE
                        WHEN ar.id IS NOT NULL THEN 'completed'
                        WHEN nr.id IS NOT NULL THEN 'no_result'
                        ELSE 'pending'
                    END AS result_status,
                    s.payload->>'games_last5' AS games_last5,
                    s.payload->>'games_last10' AS games_last10,
                    s.cohort_rank AS audit_cohort_rank
                FROM ranked_snapshots s
                LEFT JOIN audit_results ar
                    ON s.snapshot_source = 'audit'
                    AND ar.selected_date = s.selected_date
                    AND ar.market = s.market
                    AND ar.player_id = s.player_id
                LEFT JOIN audit_no_results nr
                    ON s.snapshot_source = 'audit'
                    AND nr.selected_date = s.selected_date
                    AND nr.market = s.market
                    AND nr.player_id = s.player_id
                WHERE (%s = 'all' OR s.cohort_rank <= 10)
                ORDER BY s.selected_date ASC, s.triage_score DESC
            """
            params.append(cohort)

            cursor.execute(query, params)
            audit_rows = cursor.fetchall()

    snapshots_total = len(audit_rows)
    audit_dates = sorted({row[0] for row in audit_rows if row[0] is not None})
    first_audit_date = audit_dates[0].isoformat() if audit_dates else None
    last_audit_date = audit_dates[-1].isoformat() if audit_dates else None
    completed_rows = [
        row for row in audit_rows
        if row[10] is not None
    ]
    no_result_rows = [
        row for row in audit_rows
        if row[10] is None and row[11] == 'no_result'
    ]

    def safe_average(values):
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    def build_metrics(rows):

        if not rows:
            return {
                "games": 0,
                "l5_sample": 0,
                "l10_sample": 0,
                "l5_full_window_sample": 0,
                "l5_partial_window_sample": 0,
                "l5_unknown_window_sample": 0,
                "l10_full_window_sample": 0,
                "l10_partial_window_sample": 0,
                "l10_unknown_window_sample": 0,
                "pearson_sample": 0,
                "actual_average": None,
                "score_average": None,
                "l5_prediction_average": None,
                "l10_prediction_average": None,
                "l5_mae": None,
                "l10_mae": None,
                "l5_relative_error_pct": None,
                "l10_relative_error_pct": None,
                "score_actual_pearson": None,
                "score_actual_spearman": None
            }

        actual_values = []
        score_values = []
        l5_errors = []
        l10_errors = []
        l5_predictions = []
        l10_predictions = []
        l5_window_counts = []
        l10_window_counts = []
        pairs = []

        for row in rows:

            (
                selected_date,
                player_id,
                player_name,
                team,
                opponent,
                position,
                triage_score,
                last5_avg,
                last10_avg,
                minutes_last5_avg,
                actual_value,
                result_status,
                games_last5_raw,
                games_last10_raw,
                audit_cohort_rank
            ) = row

            actual = float(actual_value)
            actual_values.append(actual)

            if last5_avg is not None:
                try:
                    l5_window_counts.append(int(games_last5_raw) if games_last5_raw is not None else None)
                except (TypeError, ValueError):
                    l5_window_counts.append(None)
                prediction_l5 = float(last5_avg)
                l5_predictions.append(prediction_l5)
                l5_errors.append(abs(actual - prediction_l5))

            if last10_avg is not None:
                try:
                    l10_window_counts.append(int(games_last10_raw) if games_last10_raw is not None else None)
                except (TypeError, ValueError):
                    l10_window_counts.append(None)
                prediction_l10 = float(last10_avg)
                l10_predictions.append(prediction_l10)
                l10_errors.append(abs(actual - prediction_l10))

            if triage_score is not None:
                score = float(triage_score)
                score_values.append(score)
                pairs.append((score, actual))

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        pearson = None

        if len(pairs) >= 2:
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            num = sum(
                (x - x_mean) * (y - y_mean)
                for x, y in pairs
            )
            den_x = sum((x - x_mean) ** 2 for x in xs)
            den_y = sum((y - y_mean) ** 2 for y in ys)
            den = (den_x * den_y) ** 0.5
            if den != 0:
                pearson = round(num / den, 4)

        spearman = spearman_correlation(pairs)

        l5_mae = safe_average(l5_errors)
        l10_mae = safe_average(l10_errors)
        l5_prediction_average = safe_average(l5_predictions)
        l10_prediction_average = safe_average(l10_predictions)

        # Erro relativo agregado: MAE / média das previsões da mesma amostra.
        l5_relative_error = (
            round(l5_mae / l5_prediction_average * 100, 2)
            if l5_mae is not None and l5_prediction_average not in (None, 0)
            else None
        )
        l10_relative_error = (
            round(l10_mae / l10_prediction_average * 100, 2)
            if l10_mae is not None and l10_prediction_average not in (None, 0)
            else None
        )

        return {
            "games": len(rows),
            "l5_sample": len(l5_errors),
            "l10_sample": len(l10_errors),
            "l5_full_window_sample": sum(value is not None and value >= 5 for value in l5_window_counts),
            "l5_partial_window_sample": sum(value is not None and value < 5 for value in l5_window_counts),
            "l5_unknown_window_sample": sum(value is None for value in l5_window_counts),
            "l10_full_window_sample": sum(value is not None and value >= 10 for value in l10_window_counts),
            "l10_partial_window_sample": sum(value is not None and value < 10 for value in l10_window_counts),
            "l10_unknown_window_sample": sum(value is None for value in l10_window_counts),
            "pearson_sample": len(pairs),
            "actual_average": safe_average(actual_values),
            "score_average": safe_average(score_values),
            "l5_prediction_average": l5_prediction_average,
            "l10_prediction_average": l10_prediction_average,
            "l5_mae": l5_mae,
            "l10_mae": l10_mae,
            "l5_relative_error_pct": l5_relative_error,
            "l10_relative_error_pct": l10_relative_error,
            "score_actual_pearson": pearson,
            "score_actual_spearman": spearman
        }

    # Faixas relativas ao próprio mercado: quintis da distribuição observada.
    # A fórmula do Score permanece intacta; aqui só mudamos a forma de agrupá-lo
    # para evitar que os mesmos limites absolutos tenham interpretações diferentes
    # entre estatísticas com distribuições distintas.
    market_score_bands = build_market_score_bands(completed_rows)
    score_band_defs = market_score_bands.get("bands", [])
    score_bands = [
        {
            "label": item["label"],
            "min": item["lower_bound"],
            "max": item["upper_bound"],
        }
        for item in score_band_defs
    ]

    bands_output = []
    for band_def in score_band_defs:
        band_rows = []
        lower = band_def.get("lower_bound")
        upper = band_def.get("upper_bound")
        for row in completed_rows:
            score = row[6]
            if score is None:
                continue
            score = float(score)
            if lower is not None and score < lower:
                continue
            if upper is not None and score >= upper:
                continue
            band_rows.append(row)

        metrics = build_metrics(band_rows)
        bands_output.append({
            "band": band_def["label"],
            "band_index": band_def["index"],
            "score_min_observed": band_def["score_min_observed"],
            "score_max_observed": band_def["score_max_observed"],
            "share_pct": band_def["share_pct"],
            **metrics
        })


    # Análise de jogadores semelhantes desativada.
    # Não integra os indicadores comerciais da auditoria e não deve consumir
    # processamento no endpoint principal de histórico.
    similarity_analyses = {}
    similarity_analysis = {}

    overall_metrics = build_metrics(completed_rows)
    model_comparison = build_model_comparison(completed_rows)

    # Avalia o ranking diário real: top 10 por data versus demais candidatos.
    top10_vs_rest = build_top10_vs_rest_metrics(completed_rows)

    daily_output = []
    audit_rows_by_date = {}
    completed_by_date = {}

    for row in audit_rows:
        d_str = row[0].isoformat() if row[0] else None
        if d_str:
            audit_rows_by_date.setdefault(d_str, []).append(row)
            if row[10] is not None:
                completed_by_date.setdefault(d_str, []).append(row)

    for d_str in sorted(audit_rows_by_date.keys()):
        date_rows = audit_rows_by_date[d_str]
        date_completed = [row for row in date_rows if row[10] is not None]
        date_no_result = [row for row in date_rows if row[10] is None and row[11] == "no_result"]
        date_pending = max(len(date_rows) - len(date_completed) - len(date_no_result), 0)
        date_resolved = len(date_completed) + len(date_no_result)
        date_metrics = build_metrics(completed_by_date.get(d_str, []))
        daily_output.append({
            "date": d_str,
            "snapshots_total": len(date_rows),
            "completed_results": len(date_completed),
            "no_result_results": len(date_no_result),
            "pending_results": date_pending,
            "completion_rate_pct": round(date_resolved / len(date_rows) * 100, 2) if date_rows else 0,
            **date_metrics
        })

    completed_count = len(completed_rows)
    no_result_count = len(no_result_rows)
    pending_rows = [
        row for row in audit_rows
        if row[10] is None and row[11] == 'pending'
    ]
    pending_count = len(pending_rows)

    def period_for_rows(rows):
        dates = sorted({row[0] for row in rows if row[0] is not None})
        return {
            "first_date": dates[0].isoformat() if dates else None,
            "last_date": dates[-1].isoformat() if dates else None
        }

    resolved_count = completed_count + no_result_count
    completion_rate = (
        round(resolved_count / snapshots_total * 100, 2)
        if snapshots_total else 0
    )

    return {
        "status": "ok",
        "market": market,
        "market_label": MARKETS[market]["label"],
        "cohort": cohort,
        "cohort_label": "Top 10 exibidos" if cohort == "top10" else "Todos os elegíveis",
        "summary": {
            "snapshots_total": snapshots_total,
            "completed_results": completed_count,
            "no_result_results": no_result_count,
            "pending_results": pending_count,
            "completion_rate_pct": completion_rate,
            **overall_metrics
        },
        "periods": {
            "snapshots": period_for_rows(audit_rows),
            "completed": period_for_rows(completed_rows),
            "no_result": period_for_rows(no_result_rows),
            "pending": period_for_rows(pending_rows)
        },
        "score_bands": bands_output,
        "score_band_reference": market_score_bands,
        "model_comparison": model_comparison,
        "similarity_analysis": similarity_analysis,
        "similarity_analyses": similarity_analyses,
        "ranking_quality": top10_vs_rest,
        "daily": daily_output,
        "data_source_policy": "Base oficial independente: somente audit_snapshots, audit_results e audit_no_results. O histórico legado operacional permanece preservado, mas não compõe estes indicadores.",
        "method_note": (
            "Precisão: MAE L5/L10, com amostras completas e parciais identificadas. "
            "Erro relativo: MAE dividido pela média das previsões correspondentes. "
            "Pearson mede associação linear e Spearman associação monotônica do ranking. "
            "Top 10 versus demais compara também o resultado ajustado pela média L10. "
            "Faixas de Score são específicas do mercado e do recorte, baseadas em quintis da distribuição observada; não são calibração probabilística. "
            "Registros sem resultado (ex.: DNP) e pendentes não entram nas métricas; "
            "a cobertura considera concluídos + sem resultado; Score original não é alterado."
        )
    }




@app.get("/context-validation")
def context_validation(
    market: str = Query(default="assists"),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    cohort: str = Query(default="all"),
):
    """Audita referências defensivas pré-jogo capturadas no snapshot independente.

    É deliberadamente exploratório: não altera Score, Gap ou snapshots históricos.
    """
    if market not in ACTIVE_MARKETS:
        raise HTTPException(status_code=400, detail="Mercado inválido ou inativo na auditoria.")
    if cohort not in {"all", "top10"}:
        raise HTTPException(status_code=400, detail="População de auditoria inválida.")

    try:
        parsed_from = datetime.strptime(date_from, "%Y-%m-%d").date() if date_from else audit_metrics_start_date()
        parsed_to = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else audit_history_cutoff_date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Período de datas inválido.")

    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    query = """
        WITH snapshot_source AS (
            SELECT
                s.selected_date,
                s.player_id,
                s.player_name,
                s.triage_score,
                s.last10_avg,
                s.audit_rank,
                s.payload,
                ar.actual_value,
                ROW_NUMBER() OVER (
                    PARTITION BY s.selected_date, s.market
                    ORDER BY s.audit_rank ASC, s.player_id ASC
                ) AS cohort_rank
            FROM audit_snapshots s
            LEFT JOIN audit_results ar
              ON ar.selected_date = s.selected_date
             AND ar.market = s.market
             AND ar.player_id = s.player_id
            WHERE s.market = %s
              AND s.selected_date >= %s
              AND s.selected_date <= %s
        )
        SELECT
            selected_date,
            player_id,
            player_name,
            last10_avg,
            actual_value,
            payload->>'opponent_position_minutes_filtered_defense_l10_avg',
            payload->>'opponent_position_minute_band_defense_l10_avg',
            payload->>'opponent_position_equal_game_defense_l10_avg',
            payload->>'opponent_position_minutes_filtered_defense_l10_games',
            payload->>'opponent_position_minute_band_defense_l10_games',
            payload->>'opponent_position_equal_game_defense_l10_games',
            payload->>'minutes_last5_avg'
        FROM snapshot_source
        WHERE actual_value IS NOT NULL
          AND (%s = 'all' OR cohort_rank <= 10)
        ORDER BY selected_date ASC, player_name ASC
    """
    params = [market, parsed_from, parsed_to, cohort]

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    records = []
    for row in rows:
        (
            selected_date,
            player_id,
            player_name,
            player_l10_avg,
            actual,
            minutes_filtered,
            minute_band,
            equal_game,
            minutes_filtered_games,
            minute_band_games,
            equal_game_games,
            minutes_last5_avg,
        ) = row
        records.append({
            "date": selected_date.isoformat() if selected_date else None,
            "player_id": str(player_id),
            "player": player_name,
            "player_l10_avg": player_l10_avg,
            "actual": actual,
            "opponent_position_minutes_filtered_defense_l10_avg": minutes_filtered,
            "opponent_position_minute_band_defense_l10_avg": minute_band,
            "opponent_position_equal_game_defense_l10_avg": equal_game,
            "minutes_filtered_games": int(minutes_filtered_games or 0),
            "minute_band_games": int(minute_band_games or 0),
            "equal_game_games": int(equal_game_games or 0),
            "minutes_last5_avg": minutes_last5_avg,
        })

    dates = sorted({item["date"] for item in records if item["date"]})
    metrics = build_context_reference_metrics(records)
    metrics["distinct_dates"] = len(dates)
    metrics["minimum_dates_for_temporal_validation"] = 14
    metrics["temporal_status"] = "ready" if len(dates) >= 14 else "collecting"

    return {
        "status": "ok",
        "market": market,
        "cohort": cohort,
        "date_from": parsed_from.isoformat(),
        "date_to": parsed_to.isoformat(),
        "metrics": metrics,
        "coverage": {
            "records_completed": len(records),
            "distinct_dates": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
        },
        "method_note": (
            "As referências de minutagem são comparadas com a média L10 do próprio "
            "jogador somente para diagnóstico exploratório. A média L10 permanece "
            "a referência individual; nenhuma dessas métricas altera o triage_score."
        ),
    }


@app.get("/teammate-split-validation")
def teammate_split_validation(
    market: str = Query(default="assists"),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    cohort: str = Query(default="all"),
):
    """Audita a associação entre splits históricos de companheiros e erro L10."""
    if market not in ACTIVE_MARKETS:
        raise HTTPException(status_code=400, detail="Mercado inválido ou inativo na auditoria.")
    if cohort not in {"all", "top10"}:
        raise HTTPException(status_code=400, detail="População de auditoria inválida.")
    try:
        parsed_from = datetime.strptime(date_from, "%Y-%m-%d").date() if date_from else audit_metrics_start_date()
        parsed_to = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else audit_history_cutoff_date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Período de datas inválido.")
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    query = """
        WITH snapshot_source AS (
            SELECT
                s.selected_date,
                s.player_id,
                s.player_name,
                s.last10_avg,
                s.audit_rank,
                s.payload,
                ar.actual_value,
                ROW_NUMBER() OVER (
                    PARTITION BY s.selected_date, s.market
                    ORDER BY s.audit_rank ASC, s.player_id ASC
                ) AS cohort_rank
            FROM audit_snapshots s
            LEFT JOIN audit_results ar
              ON ar.selected_date = s.selected_date
             AND ar.market = s.market
             AND ar.player_id = s.player_id
            WHERE s.market = %s
              AND s.selected_date >= %s
              AND s.selected_date <= %s
        )
        SELECT selected_date, player_id, player_name, last10_avg,
               actual_value, payload
        FROM snapshot_source
        WHERE actual_value IS NOT NULL
          AND (%s = 'all' OR cohort_rank <= 10)
        ORDER BY selected_date ASC, player_name ASC
    """
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, [market, parsed_from, parsed_to, cohort])
            rows = cursor.fetchall()

    stat_key = {
        "points": "points",
        "assists": "assists",
        "rebounds": "rebounds",
        "threes": "three_pa",
        "two_pa": "two_pa",
        "fgm": "fgm",
    }[market]
    records = []
    for selected_date, player_id, player_name, last10_avg, actual, payload in rows:
        payload = payload if isinstance(payload, dict) else {}
        records.append({
            "date": selected_date.isoformat() if selected_date else None,
            "player_id": str(player_id),
            "player": player_name,
            "player_l10_avg": last10_avg,
            "actual": actual,
            "teammate_context": payload.get("teammate_context") or {},
        })

    metrics = build_teammate_split_metrics(records, stat_key)
    dates = sorted({item["date"] for item in records if item.get("date")})
    return {
        "status": "ok",
        "market": market,
        "market_label": MARKETS[market]["label"],
        "cohort": cohort,
        "date_from": parsed_from.isoformat(),
        "date_to": parsed_to.isoformat(),
        "metrics": metrics,
        "coverage": {
            "records_completed": len(records),
            "distinct_dates": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
        },
    }


@app.get("/gap-validation")
def gap_validation(market: str = Query(default="assists"), score_band: str = Query(default="all"), date_from: Optional[str] = Query(default=None), date_to: Optional[str] = Query(default=None), cohort: str = Query(default="all")):
    """Audita se a defesa permite mais ou menos produção por posição que a média recente da liga."""
    if market not in ACTIVE_MARKETS:
        raise HTTPException(status_code=400, detail="Mercado inválido ou inativo na auditoria.")
    if cohort not in {"top10", "all"}:
        raise HTTPException(status_code=400, detail="População de auditoria inválida.")
    valid_bands = {"all", "7+", "6-6.99", "5-5.99", "4-4.99", "<4"}
    if score_band not in valid_bands:
        raise HTTPException(status_code=400, detail="Faixa de Score inválida.")
    try:
        parsed_from = datetime.strptime(date_from, "%Y-%m-%d").date() if date_from else None
        parsed_to = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Período de datas inválido.")
    # Sem datas: métricas acumuladas da coorte validada desde 19/11.
    if not date_from and not date_to:
        parsed_from = audit_metrics_start_date()
        parsed_to = audit_history_cutoff_date()

    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    query = """
        WITH snapshot_source AS (
            SELECT
                selected_date, market, player_id, player_name, team, opponent,
                position, triage_score, last5_avg, last10_avg, payload,
                audit_rank AS display_rank, 'audit' AS snapshot_source
            FROM audit_snapshots
            WHERE market = %s

        ), snapshot_cohort AS (
            SELECT *,
                COUNT(*) FILTER (WHERE display_rank IS NULL)
                    OVER (PARTITION BY selected_date, market) AS missing_rank_count
            FROM snapshot_source
        ), ranked_snapshots AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY selected_date, market ORDER BY
                CASE WHEN missing_rank_count = 0 THEN display_rank END ASC NULLS LAST,
                triage_score DESC NULLS LAST, player_id ASC) AS cohort_rank
            FROM snapshot_cohort
        )
        SELECT s.selected_date, s.player_id, s.player_name, s.team, s.opponent, s.position, s.triage_score,
               s.last5_avg, s.last10_avg,
               s.payload->>'opponent_position_allowed_l5_avg',
               s.payload->>'opponent_position_allowed_l10_avg',
               s.payload->>'league_position_allowed_l5_avg',
               s.payload->>'league_position_allowed_l10_avg',
               s.payload->>'opponent_position_defense_games_l5',
               s.payload->>'opponent_position_defense_games_l10',
               s.payload->>'league_position_allowed_l5_games',
               s.payload->>'league_position_allowed_l10_games',
               s.payload->>'opponent_position_equal_game_reference_version',
               ar.actual_value
        FROM ranked_snapshots s
        LEFT JOIN audit_results ar
          ON s.snapshot_source = 'audit'
         AND ar.selected_date = s.selected_date AND ar.market = s.market AND ar.player_id = s.player_id
        WHERE (%s = 'all' OR s.cohort_rank <= 10)
          AND ar.id IS NOT NULL
    """
    params = [market, cohort]
    limits = {"7+": (7.0, None), "6-6.99": (6.0, 7.0), "5-5.99": (5.0, 6.0), "4-4.99": (4.0, 5.0), "<4": (None, 4.0)}
    if score_band != "all":
        low, high = limits[score_band]
        if low is not None:
            query += " AND s.triage_score >= %s"; params.append(low)
        if high is not None:
            query += " AND s.triage_score < %s"; params.append(high)
    if parsed_from:
        query += " AND s.selected_date >= %s"; params.append(parsed_from)
    if parsed_to:
        query += " AND s.selected_date <= %s"; params.append(parsed_to)
    query += " ORDER BY s.selected_date DESC, s.triage_score DESC"
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    def number(value):
        try:
            result = float(value) if value is not None else None
            return result if result is not None and result == result and abs(result) != float("inf") else None
        except (TypeError, ValueError):
            return None

    def classify(allowed, league_average):
        if allowed is None or league_average is None:
            return "Sem referência", None
        if league_average == 0:
            return ("Acima", None) if allowed > 0 else ("Próximo", 0.0)
        pct = (allowed - league_average) / abs(league_average) * 100
        if pct > 10:
            return "Acima", round(pct, 1)
        if pct < -10:
            return "Abaixo", round(pct, 1)
        return "Próximo", round(pct, 1)

    detailed = []
    legacy_records = 0
    for row in rows:
        (game_date, player_id, player, team, opponent, position, score, l5, l10,
         allowed5_raw, allowed10_raw, league5_raw, league10_raw, games5_raw, games10_raw,
         league_games5_raw, league_games10_raw, reference_version, actual_raw) = row
        actual = number(actual_raw)
        allowed5, allowed10 = number(allowed5_raw), number(allowed10_raw)
        league5, league10 = number(league5_raw), number(league10_raw)
        games5, games10 = int(number(games5_raw) or 0), int(number(games10_raw) or 0)
        league_games5 = int(number(league_games5_raw) or 0)
        league_games10 = int(number(league_games10_raw) or 0)

        if int(number(reference_version) or 0) < 2:
            legacy_records += 1
            allowed5 = allowed10 = league5 = league10 = None
            games5 = games10 = league_games5 = league_games10 = 0

        class5, pct5 = classify(allowed5, league5)
        class10, pct10 = classify(allowed10, league10)

        def period(label, allowed, league_average, pct, games, league_games):
            return {
                "classification": label,
                "defense_allowed": allowed,
                "league_average": league_average,
                "difference": round(allowed - league_average, 2) if allowed is not None and league_average is not None else None,
                "difference_pct": pct,
                "games": games,
                "league_games": league_games
            }

        detailed.append({
            "date": game_date.isoformat() if game_date else None,
            "player_id": str(player_id), "player": player, "team": team, "opponent": opponent,
            "position": position, "score": number(score), "actual": actual,
            "player_l5_avg": number(l5), "player_l10_avg": number(l10),
            "l5": period(class5, allowed5, league5, pct5, games5, league_games5),
            "l10": period(class10, allowed10, league10, pct10, games10, league_games10)
        })

    def summarize(period_key):
        output = {}
        for label in ("Acima", "Próximo", "Abaixo", "Sem referência"):
            group = [item for item in detailed if item[period_key]["classification"] == label]
            output[label] = build_gap_group_metrics(group, period_key)
        return output

    return {
        "status": "ok", "market": market, "market_label": MARKETS[market]["label"],
        "cohort": cohort, "cohort_label": "Top 10 exibidos" if cohort == "top10" else "Todos os elegíveis",
        "score_band": score_band, "completed_records": len(detailed),
        "legacy_records_without_new_reference": legacy_records,
        "data_source_policy": "Base oficial independente: somente audit_snapshots, audit_results e audit_no_results. O histórico legado operacional permanece preservado, mas não compõe estes indicadores.",
        "classification_margin_pct": 10,
        "periods": {"l5": summarize("l5"), "l10": summarize("l10")},
        "records": detailed,
        "method_note": "Contexto defensivo = total da estatística cedida aos jogadores da posição por partida, comparado à média da liga para a mesma posição e período. Acima significa que a defesa permite mais de 10% que a liga; Abaixo, menos de 10%; Próximo, dentro de ±10%. A média real exibida é descritiva e não demonstra causalidade nem capacidade preditiva da classificação. Snapshots antigos não possuem o benchmark da liga e ficam sem referência."
    }


# Página inicial do Scanner, servida pelo mesmo domínio da API no Render.
INDEX_FILE = Path(__file__).resolve().parent.parent / "index.html"


@app.get("/", include_in_schema=False)
def home():
    if not INDEX_FILE.is_file():
        raise HTTPException(status_code=404, detail="Arquivo index.html não encontrado.")
    return FileResponse(INDEX_FILE, media_type="text/html")


# ============================================================
# CENTRAL DE CONTEXTO NBA (MÓDULO INDEPENDENTE)
# Cache de summaries compartilhado com o Scanner para evitar requisições duplicadas.
# ============================================================
# A Central usa armazenamento próprio e não participa do fluxo de
# candidatos, do triage_score ou das tabelas da Auditoria.
from app.context_center import create_context_router

app.include_router(
    create_context_router(
        get_database_connection,
        get_team_defense_history_fresh,
        NBA_TEAM_ABBREVIATIONS,
        require_admin_api_key
    )
)