import logging
import time
import re
import os
import threading

from datetime import datetime, date
from zoneinfo import ZoneInfo
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from pathlib import Path

import requests
import psycopg
from psycopg.types.json import Jsonb

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nba_props_scanner")

app = FastAPI(
    title="NBA Props Scanner",
    description="API de triagem estatística NBA via ESPN.",
    version="3.16.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BANCO DE DADOS - HISTÓRICO IMUTÁVEL DE CANDIDATOS
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


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

        for competitor in competition.get(
            "competitors",
            []
        ):

            competitor_team = (
                competitor.get("team", {})
                .get("abbreviation")
            )

            if competitor_team != team_abbr:
                opponent = competitor_team
                break

        if not opponent:
            continue

        games.append({
            "game_id": str(
                event.get("id")
            ),
            "date": event.get("date"),
            "opponent": opponent
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
    target_date=None
):

    if target_date is None:
        target_date = date.today()

    games = [
        game
        for game in get_team_regular_season_schedule(
            team_abbr.upper(),
            season_year(target_date)
        )
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


def fetch_espn_games(
    selected_date: str
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

            item = {
                "abbr": team.get(
                    "abbreviation",
                    ""
                ),
                "name": team.get(
                    "displayName",
                    ""
                )
            }

            if competitor.get("homeAway") == "home":
                home = item
            elif competitor.get("homeAway") == "away":
                away = item

        if home and away:

            # A ESPN fornece o início em UTC; a interface exibe horário de Brasília.
            event_date = event.get("date")
            start_time_brt = None

            if event_date:
                try:
                    start_utc = datetime.fromisoformat(
                        event_date.replace("Z", "+00:00")
                    )
                    start_time_brt = start_utc.astimezone(
                        ZoneInfo("America/Sao_Paulo")
                    ).strftime("%H:%M")
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

    schedule = fetch_espn_games(
        selected_date
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

    data = request_json(
        GAMELOG_URL.format(
            athlete=athlete_id
        ),
        {"season": year}
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

    ordered = sorted(
        games.values(),
        key=lambda game: game["date"],
        reverse=True
    )

    return tuple(ordered)


def extract_player_game_from_summary(
    summary,
    player_id
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

                if raw_stats is None:
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


@lru_cache(maxsize=12)
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
        "unique_games": len(summaries)
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

    consistency = (
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
        + consistency * 0.20
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
    limit
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

    # Primeiro buscamos os históricos dos jogadores. Isso permite
    # identificar quem realmente pode virar candidato antes de
    # consultar o histórico defensivo dos adversários.
    stage_start = time.perf_counter()
    player_logs = {}
    successful_logs = 0
    errors = 0

    def fetch_player(player):
        logs = get_athlete_logs(
            player["id"],
            year
        )
        return player, logs

    with ThreadPoolExecutor(
        max_workers=12
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
                player, logs = future.result()
                successful_logs += 1
                player_logs[player["id"]] = (player, logs)

            except Exception as error:
                errors += 1
                logger.warning(
                    "Falha ao consultar jogador: %s",
                    error
                )

    player_logs_elapsed = time.perf_counter() - stage_start

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

    league_baseline = None

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        futures = [
            executor.submit(fetch_team_defense, team)
            for team in eligible_opponents
        ]
        league_future = (
            executor.submit(
                get_league_position_defense_baseline,
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
                league_baseline = league_future.result()
            except Exception as error:
                logger.warning(
                    "Falha ao calcular benchmark defensivo da liga: %s",
                    error
                )

    defense_elapsed = time.perf_counter() - stage_start

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

    return {
        "date": selected_date,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "count": min(len(results), final_limit),
        "source": "ESPN",
        "season": season_for(target_date),
        "teams_found": roster_data["teams_found"],
        "players_found": len(players),
        "players_consulted": successful_logs,
        "players_with_errors": errors,
        "updated_at": datetime.now().isoformat(),
        "candidates": results[:final_limit]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NBA Props Scanner",
        "version": "3.15.0",
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

    return fetch_espn_games(selected_date)


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


@app.get("/collect-results")
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
        selected_date
    )

    completed_games = {}
    completed_game_ids = []

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

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    player_id,
                    player_name,
                    team,
                    opponent,
                    payload
                FROM candidate_snapshots
                WHERE
                    selected_date = %s
                    AND market = %s
                ORDER BY player_name
                """,
                (target_date, market)
            )
            snapshots = cursor.fetchall()

    if not snapshots:
        return {
            "date": selected_date,
            "market": market,
            "status": "no_snapshots",
            "results_saved": 0
        }

    summary_cache = {}

    def fetch_summary(game_id):
        return (
            game_id,
            get_game_summary(game_id)
        )

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        futures = [
            executor.submit(
                fetch_summary,
                game_id
            )
            for game_id in completed_game_ids
        ]

        for future in as_completed(futures):
            try:
                game_id, summary = future.result()
                summary_cache[game_id] = summary
            except Exception:
                pass

    results_to_save = []
    pending = 0

    for snapshot in snapshots:

        (
            player_id,
            player_name,
            team,
            opponent,
            payload
        ) = snapshot

        game_id = completed_games.get(
            (
                (team or "").upper(),
                (opponent or "").upper()
            )
        )

        if not game_id:
            pending += 1
            continue

        summary = summary_cache.get(game_id)

        if not summary:
            pending += 1
            continue

        player_game = (
            extract_player_game_from_summary(
                summary,
                player_id
            )
        )

        if not player_game:
            pending += 1
            continue

        minutes = player_game.get("minutes")

        if minutes is None or minutes <= 0:
            pending += 1
            continue

        actual = player_game.get(actual_stat_key)

        if actual is None:
            pending += 1
            continue

        results_to_save.append((
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

    inserted = 0

    if results_to_save:
        with get_database_connection() as connection:
            with connection.cursor() as cursor:
                for row in results_to_save:
                    cursor.execute(
                        """
                        INSERT INTO candidate_results (
                            selected_date,
                            market,
                            player_id,
                            player_name,
                            team,
                            opponent,
                            actual_value,
                            game_id,
                            payload
                        )
                        VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        ON CONFLICT (
                            selected_date,
                            market,
                            player_id
                        ) DO NOTHING
                        RETURNING id
                        """,
                        row
                    )
                    if cursor.fetchone():
                        inserted += 1

    return {
        "date": selected_date,
        "market": market,
        "status": "processed",
        "results_saved_this_run": inserted,
        "pending": pending
    }


@app.get("/collect-pending-results")
def collect_pending_results(
    max_combinations: int = Query(default=100, ge=1, le=200)
):
    """
    Coleta automaticamente resultados de todos os snapshots pendentes.

    Procura combinações data/mercado que ainda não possuem resultado,
    tenta processá-las pelo mesmo fluxo usado pela coleta manual e
    deixa pendente apenas o que ainda não pôde ser resolvido.
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
                SELECT DISTINCT
                    s.selected_date,
                    s.market
                FROM candidate_snapshots s
                LEFT JOIN candidate_results r
                    ON r.selected_date = s.selected_date
                    AND r.market = s.market
                    AND r.player_id = s.player_id
                WHERE
                    r.id IS NULL
                    AND s.selected_date <= %s
                    AND s.market = ANY(%s)
                ORDER BY
                    s.selected_date ASC,
                    s.market ASC
                LIMIT %s
                """,
                (
                    today,
                    list(ACTIVE_MARKETS),
                    max_combinations
                )
            )
            pending_combinations = cursor.fetchall()

    processed = []
    total_saved = 0
    total_pending = 0
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

            processed.append({
                "date": selected_date.isoformat(),
                "market": market,
                "results_saved": saved,
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
        "pending_after_processing": total_pending,
        "errors": total_errors,
        "processed": processed
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
    - calibração: quanto cada faixa de score realmente corresponde
      à produção dos jogadores.
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

    if not DATABASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados não configurado."
        )

    with get_database_connection() as connection:
        with connection.cursor() as cursor:

            query = """
                WITH snapshot_cohort AS (
                    SELECT
                        s.selected_date,
                        s.market,
                        s.player_id,
                        s.player_name,
                        s.team,
                        s.opponent,
                        s.position,
                        s.triage_score,
                        s.last5_avg,
                        s.last10_avg,
                        s.minutes_last5_avg,
                        s.display_rank,
                        COUNT(*) FILTER (
                            WHERE s.display_rank IS NULL
                        ) OVER (
                            PARTITION BY s.selected_date, s.market
                        ) AS missing_rank_count
                    FROM candidate_snapshots s
                    WHERE s.market = %s
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
                        ROW_NUMBER() OVER (
                            PARTITION BY selected_date, market
                            ORDER BY
                                CASE
                                    WHEN missing_rank_count = 0
                                    THEN display_rank
                                END ASC NULLS LAST,
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
                    r.actual_value
                FROM ranked_snapshots s
                LEFT JOIN candidate_results r
                    ON r.selected_date = s.selected_date
                    AND r.market = s.market
                    AND r.player_id = s.player_id
                WHERE (%s = 'all' OR s.cohort_rank <= 10)
                ORDER BY s.selected_date ASC, s.triage_score DESC
            """
            params.append(cohort)

            cursor.execute(query, params)
            audit_rows = cursor.fetchall()

    snapshots_total = len(audit_rows)
    completed_rows = [
        row for row in audit_rows
        if row[10] is not None
    ]

    def safe_average(values):
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    def build_metrics(rows):

        if not rows:
            return {
                "games": 0,
                "actual_average": None,
                "score_average": None,
                "l5_prediction_average": None,
                "l10_prediction_average": None,
                "l5_mae": None,
                "l10_mae": None,
                "l5_relative_error_pct": None,
                "l10_relative_error_pct": None,
                "score_actual_pearson": None
            }

        actual_values = []
        score_values = []
        l5_errors = []
        l10_errors = []
        l5_predictions = []
        l10_predictions = []
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
                actual_value
            ) = row

            actual = float(actual_value)
            actual_values.append(actual)

            if last5_avg is not None:
                prediction_l5 = float(last5_avg)
                l5_predictions.append(prediction_l5)
                l5_errors.append(abs(actual - prediction_l5))

            if last10_avg is not None:
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
            "actual_average": safe_average(actual_values),
            "score_average": safe_average(score_values),
            "l5_prediction_average": l5_prediction_average,
            "l10_prediction_average": l10_prediction_average,
            "l5_mae": l5_mae,
            "l10_mae": l10_mae,
            "l5_relative_error_pct": l5_relative_error,
            "l10_relative_error_pct": l10_relative_error,
            "score_actual_pearson": pearson
        }

    # Faixas de score calibradas
    score_bands = [
        {"label": "7+", "min": 7.0, "max": None},
        {"label": "6-6.99", "min": 6.0, "max": 7.0},
        {"label": "5-5.99", "min": 5.0, "max": 6.0},
        {"label": "4-4.99", "min": 4.0, "max": 5.0},
        {"label": "<4", "min": None, "max": 4.0}
    ]

    bands_output = []

    for band in score_bands:

        band_rows = []

        for row in completed_rows:

            score = row[6]

            if score is None:
                continue

            score = float(score)

            if band["min"] is not None and score < band["min"]:
                continue
            if band["max"] is not None and score >= band["max"]:
                continue

            band_rows.append(row)

        metrics = build_metrics(band_rows)
        bands_output.append({
            "band": band["label"],
            **metrics
        })


    # Investigação de perfis semelhantes para todas as faixas de score.
    # A faixa escolhida é comparada com registros concluídos de outras faixas,
    # priorizando outros jogadores da mesma posição e L5/L10/minutos próximos.
    def normalize_audit_position(value):
        raw = str(value or "").strip().upper()
        if raw in ("G", "PG", "SG", "GUARD"):
            return "G"
        if raw in ("F", "SF", "PF", "FORWARD"):
            return "F"
        if raw in ("C", "CENTER"):
            return "C"
        return "OTHER" if raw else "N/D"

    def audit_minutes_band(value):
        if value is None:
            return "N/D"
        minutes = float(value)
        if minutes < 24:
            return "<24 min"
        if minutes < 30:
            return "24–29,9 min"
        return "30+ min"

    feature_indexes = (7, 8, 9)  # L5, L10 e minutos médios.
    feature_scales = {}
    for feature_index in feature_indexes:
        values = [
            float(row[feature_index])
            for row in completed_rows
            if row[feature_index] is not None
        ]
        if values:
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            feature_scales[feature_index] = variance ** 0.5 or 1.0
        else:
            feature_scales[feature_index] = 1.0

    similarity_analyses = {}

    for score_band in score_bands:
        band_label = score_band["label"]

        def row_in_band(row, band):
            score = row[6]
            if score is None:
                return False
            score = float(score)
            if band["min"] is not None and score < band["min"]:
                return False
            if band["max"] is not None and score >= band["max"]:
                return False
            return True

        target_rows_for_band = [
            row for row in completed_rows if row_in_band(row, score_band)
        ]
        other_candidates = [
            row for row in completed_rows
            if row[6] is not None and not row_in_band(row, score_band)
        ]

        similarity_groups = {}
        matched_target_rows = []

        for target in target_rows_for_band:
            target_position = normalize_audit_position(target[5])
            target_minutes_band = audit_minutes_band(target[9])
            group_key = (target_position, target_minutes_band)
            group = similarity_groups.setdefault(group_key, {
                "targets": [],
                "peers": {}
            })

            target_features = [target[index] for index in feature_indexes]
            if any(value is None for value in target_features):
                continue

            target_player_id = str(target[1])
            ranked_peers = []

            for candidate in other_candidates:
                if str(candidate[1]) == target_player_id:
                    continue
                if normalize_audit_position(candidate[5]) != target_position:
                    continue

                candidate_features = [candidate[index] for index in feature_indexes]
                if any(value is None for value in candidate_features):
                    continue

                squared_distances = [
                    (
                        (float(target_value) - float(candidate_value))
                        / feature_scales[feature_index]
                    ) ** 2
                    for feature_index, target_value, candidate_value
                    in zip(feature_indexes, target_features, candidate_features)
                ]
                distance = (sum(squared_distances) / len(squared_distances)) ** 0.5

                if distance <= 1.25:
                    ranked_peers.append((distance, candidate))

            ranked_peers.sort(key=lambda item: item[0])
            if not ranked_peers:
                continue

            group["targets"].append(target)
            matched_target_rows.append(target)

            for distance, peer in ranked_peers[:3]:
                peer_key = (peer[0], str(peer[1]))
                if peer_key not in group["peers"]:
                    group["peers"][peer_key] = {
                        "row": peer,
                        "distance": distance
                    }

        similarity_cohorts = []
        matched_peer_rows_by_key = {}

        for (position_group, minutes_group), group in sorted(similarity_groups.items()):
            target_rows = group["targets"]
            if not target_rows:
                continue

            peer_entries = list(group["peers"].values())
            peer_rows = [entry["row"] for entry in peer_entries]
            for peer in peer_rows:
                peer_key = (peer[0], str(peer[1]))
                matched_peer_rows_by_key.setdefault(peer_key, peer)

            target_average = safe_average([float(row[10]) for row in target_rows])
            peer_average = safe_average([float(row[10]) for row in peer_rows])
            target_players = len({str(row[1]) for row in target_rows})
            peer_players = len({str(row[1]) for row in peer_rows})

            similarity_cohorts.append({
                "position": position_group,
                "minutes_band": minutes_group,
                "target_records": len(target_rows),
                "target_players": target_players,
                "target_actual_average": target_average,
                "similar_records": len(peer_rows),
                "similar_players": peer_players,
                "similar_actual_average": peer_average,
                "actual_difference": (
                    round(target_average - peer_average, 2)
                    if target_average is not None and peer_average is not None
                    else None
                ),
                "actual_difference_pct": (
                    round((target_average - peer_average) / abs(peer_average) * 100, 2)
                    if target_average is not None and peer_average not in (None, 0)
                    else None
                ),
                "average_similarity_distance": safe_average(
                    [entry["distance"] for entry in peer_entries]
                ),
                "sample_status": (
                    "Volume maior"
                    if len(target_rows) >= 30 and target_players >= 15
                    and len(peer_rows) >= 30 and peer_players >= 15
                    else "Em acompanhamento"
                    if len(target_rows) >= 10 and target_players >= 5
                    and len(peer_rows) >= 20 and peer_players >= 8
                    else "Amostra reduzida"
                )
            })

        matched_peer_rows = list(matched_peer_rows_by_key.values())
        target_average = safe_average([float(row[10]) for row in matched_target_rows])
        peer_average = safe_average([float(row[10]) for row in matched_peer_rows])
        target_players = len({str(row[1]) for row in matched_target_rows})
        peer_players = len({str(row[1]) for row in matched_peer_rows})

        similarity_analyses[band_label] = {
            "target_band": band_label,
            "target_count": len(matched_target_rows),
            "target_players": target_players,
            "target_actual_average": target_average,
            "similar_count": len(matched_peer_rows),
            "similar_players": peer_players,
            "similar_actual_average": peer_average,
            "actual_difference": (
                round(target_average - peer_average, 2)
                if target_average is not None and peer_average is not None
                else None
            ),
            "actual_difference_pct": (
                round((target_average - peer_average) / abs(peer_average) * 100, 2)
                if target_average is not None and peer_average not in (None, 0)
                else None
            ),
            "sample_status": (
                "Volume maior"
                if len(matched_target_rows) >= 30 and target_players >= 15
                and len(matched_peer_rows) >= 30 and peer_players >= 15
                else "Em acompanhamento"
                if len(matched_target_rows) >= 10 and target_players >= 5
                and len(matched_peer_rows) >= 20 and peer_players >= 8
                else "Amostra reduzida"
            ),
            "cohorts": similarity_cohorts,
            "method_note": (
                "Compara registros concluídos da faixa selecionada com até três "
                "registros de outros jogadores fora dessa faixa, da mesma posição "
                "e com L5, L10 e minutos médios mais próximos. As variáveis são "
                "padronizadas pelo desvio-padrão do mercado; pares com distância "
                "média padronizada acima de 1,25 são descartados. Análise exploratória."
            )
        }

    # Mantém compatibilidade com consumidores da versão anterior da API.
    similarity_analysis = similarity_analyses.get("6-6.99", {})
    overall_metrics = build_metrics(completed_rows)

    # Separação entre a faixa de maior score e a de menor score.
    high_band = next((b for b in bands_output if b["band"] == "7+"), None)
    low_band = next((b for b in bands_output if b["band"] == "<4"), None)
    high_low = {
        "high_score_band": high_band,
        "low_score_band": low_band,
        "actual_average_difference": (
            round(high_band["actual_average"] - low_band["actual_average"], 2)
            if high_band and low_band
            and high_band["actual_average"] is not None
            and low_band["actual_average"] is not None
            else None
        ),
        "high_vs_low_pct": (
            round(
                (high_band["actual_average"] - low_band["actual_average"])
                / low_band["actual_average"] * 100, 2
            )
            if high_band and low_band
            and high_band["actual_average"] is not None
            and low_band["actual_average"] not in (None, 0)
            else None
        )
    }

    by_date = {}

    for row in completed_rows:
        d_str = row[0].isoformat() if row[0] else None
        if d_str:
            by_date.setdefault(d_str, []).append(row)

    daily_output = []

    for d_str in sorted(by_date.keys()):
        daily_metrics = build_metrics(by_date[d_str])
        daily_output.append({
            "date": d_str,
            **daily_metrics
        })

    completed_count = len(completed_rows)
    pending_count = max(snapshots_total - completed_count, 0)
    completion_rate = (
        round(completed_count / snapshots_total * 100, 2)
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
            "pending_results": pending_count,
            "completion_rate_pct": completion_rate,
            **overall_metrics
        },
        "score_bands": bands_output,
        "similarity_analysis": similarity_analysis,
        "similarity_analyses": similarity_analyses,
        "ranking_quality": high_low,
        "daily": daily_output,
        "method_note": (
            "Precisão: MAE L5/L10. Erro relativo: MAE dividido pela média "
            "das previsões correspondentes, usando a mesma amostra válida. "
            "Ranking: médias reais e quantidade por faixa de score. "
            "Registros pendentes são excluídos; score original não é alterado."
        )
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
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado.")

    query = """
        WITH snapshot_cohort AS (
            SELECT s.selected_date, s.market, s.player_id, s.player_name, s.team, s.opponent, s.position,
                   s.triage_score, s.last5_avg, s.last10_avg, s.payload, s.display_rank,
                   COUNT(*) FILTER (WHERE s.display_rank IS NULL) OVER (PARTITION BY s.selected_date, s.market) AS missing_rank_count
            FROM candidate_snapshots s WHERE s.market = %s
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
               r.actual_value
        FROM ranked_snapshots s INNER JOIN candidate_results r
          ON r.selected_date = s.selected_date AND r.market = s.market AND r.player_id = s.player_id
        WHERE (%s = 'all' OR s.cohort_rank <= 10)
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
            "l5": period(class5, allowed5, league5, pct5, games5, league_games5),
            "l10": period(class10, allowed10, league10, pct10, games10, league_games10)
        })

    def summarize(period_key):
        output = {}
        for label in ("Acima", "Próximo", "Abaixo", "Sem referência"):
            group = [item for item in detailed if item[period_key]["classification"] == label]
            valid = [item for item in group if item["actual"] is not None]
            def avg(values):
                return round(sum(values) / len(values), 2) if values else None
            output[label] = {
                "count": len(group),
                "completed": len(valid),
                "defense_allowed_average": avg([x[period_key]["defense_allowed"] for x in group if x[period_key]["defense_allowed"] is not None]),
                "league_average": avg([x[period_key]["league_average"] for x in group if x[period_key]["league_average"] is not None]),
                "actual_player_average": avg([x["actual"] for x in valid]),
                "deviation_pct_average": avg([x[period_key]["difference_pct"] for x in group if x[period_key]["difference_pct"] is not None]),
                "defense_games_average": avg([x[period_key]["games"] for x in group if x[period_key]["games"] > 0]),
                "league_games_average": avg([x[period_key]["league_games"] for x in group if x[period_key]["league_games"] > 0])
            }
        return output

    return {
        "status": "ok", "market": market, "market_label": MARKETS[market]["label"],
        "cohort": cohort, "cohort_label": "Top 10 exibidos" if cohort == "top10" else "Todos os elegíveis",
        "score_band": score_band, "completed_records": len(detailed),
        "legacy_records_without_new_reference": legacy_records, "classification_margin_pct": 10,
        "periods": {"l5": summarize("l5"), "l10": summarize("l10")},
        "records": detailed,
        "method_note": "Contexto defensivo = total da estatística cedida aos jogadores da posição por partida, comparado à média da liga para a mesma posição e período. Acima significa que a defesa permite mais de 10% que a liga; Abaixo, menos de 10%; Próximo, dentro de ±10%. A média real exibida é a produção individual dos candidatos agrupados pela classificação defensiva. Snapshots antigos não possuem o benchmark da liga e ficam sem referência."
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
# ============================================================
# A Central usa armazenamento próprio e não participa do fluxo de
# candidatos, do triage_score ou das tabelas da Auditoria.
from app.context_center import create_context_router

app.include_router(
    create_context_router(
        get_database_connection,
        get_team_defense_history,
        NBA_TEAM_ABBREVIATIONS
    )
)
