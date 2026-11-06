import logging
import time
import re
import os

from datetime import datetime, date
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import psycopg
from psycopg.types.json import Jsonb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nba_props_scanner")

app = FastAPI(
    title="NBA Props Scanner",
    description="API de triagem estatística NBA via ESPN.",
    version="3.13.0"
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
                        payload JSONB NOT NULL,
                        captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT uq_candidate_snapshot
                            UNIQUE (selected_date, market, player_id)
                    )
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

    for candidate in candidates:

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
                        payload
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
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

    "attempts": {
        "index": 1,
        "label": "Tentativas totais"
    }
}


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

def request_json(
    url: str,
    params=None
) -> dict:

    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
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


def parse_optional_number(value):

    """
    Preserva a diferença entre zero real e dado ausente.
    """

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

    """
    Extrai tentativas de valores como 10-22.

    Também aceita valores numéricos quando a ESPN
    retorna apenas a quantidade tentada.
    """

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

    return {

        "fga_allowed": parse_attempts(
            stats.get(
                "fieldGoalsMade-fieldGoalsAttempted"
            )
        ),

        "three_pa_allowed": parse_attempts(
            stats.get(
                "threePointFieldGoalsMade-threePointFieldGoalsAttempted"
            )
        ),

        "rebounds_allowed": parse_number(
            stats.get("totalRebounds")
        ),

        "assists_allowed": parse_number(
            stats.get("assists")
        )
    }


# ============================================================
# AUXILIARES PARA LEITURA DE ESTATÍSTICAS POR POSIÇÃO
# ============================================================

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

    if position in (
        "G",
        "PG",
        "SG",
        "GUARD"
    ):
        return "G"

    if position in (
        "F",
        "SF",
        "PF",
        "FORWARD"
    ):
        return "F"

    if position in (
        "C",
        "CENTER"
    ):
        return "C"

    return "OTHER"


def normalize_stat_key(key):

    if key is None:
        return ""

    key = str(key).strip()

    aliases = {

        "FG":
            "fieldGoalsMade-fieldGoalsAttempted",

        "3PT":
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted",

        "REB":
            "rebounds",

        "AST":
            "assists",

        "MIN":
            "minutes",

        "PTS":
            "points"
    }

    upper_key = key.upper()

    if upper_key in aliases:
        return aliases[upper_key]

    return key


def get_player_stat_value(
    raw_stats,
    key_indexes,
    stat_key,
    attempt_stat=False
):

    """
    Recupera uma estatística individual.

    Suporta:
    - Lista de valores com índice identificado por keys.
    - Lista de valores com índice identificado por names.
    - Dicionário de estatísticas.
    - Estatísticas no formato dict com value/displayValue.

    Para minutos, também interpreta valores no formato MM:SS.
    """

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

        if value in (
            "",
            "--",
            "-",
            "N/A"
        ):
            return None

        if value is None:
            return None

        return parse_minutes(value)

    if attempt_stat:

        return parse_optional_attempts(
            value
        )

    return parse_optional_number(
        value
    )


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


# ============================================================
# JOGADORES DO ADVERSÁRIO AGRUPADOS POR POSIÇÃO
# ============================================================

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

    boxscore = summary.get(
        "boxscore",
        {}
    )

    teams = boxscore.get(
        "teams",
        []
    )

    players_blocks = boxscore.get(
        "players",
        []
    )

    defending_team = (
        defending_team or ""
    ).upper()

    opponent_team = None

    for team in teams:

        team_abbr = (
            team.get("team", {})
            .get("abbreviation")
            or ""
        ).upper()

        if (
            team_abbr
            and team_abbr != defending_team
        ):

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

        return {
            "opponent": opponent_team,
            "players": result
        }

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

            player_name = (
                athlete.get("displayName")
                or athlete.get("fullName")
                or athlete.get("shortName")
            )

            if not player_name:
                continue

            raw_stats = athlete_entry.get(
                "stats"
            )

            if raw_stats is None:

                raw_stats = athlete_entry.get(
                    "statistics"
                )

            if not raw_stats:
                continue

            position_group = get_athlete_position(
                athlete
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

            minutes = get_player_stat_value(
                raw_stats,
                key_indexes,
                "minutes"
            )

            if (
                minutes is not None
                and minutes <= 0
            ):
                continue

            if (
                minutes is None
                and all(
                    value is None
                    for value in (
                        assists,
                        rebounds,
                        fga,
                        three_pa
                    )
                )
            ):
                continue

            result[position_group].append({

                "player": player_name,

                "position": position_group,

                "minutes": minutes,

                "assists": assists,

                "rebounds": rebounds,

                "fga": fga,

                "three_pa": three_pa
            })

    return {

        "opponent": opponent_team,

        "players": result
    }


# ============================================================
# AGREGA ESTATÍSTICAS POR POSIÇÃO EM UMA PARTIDA
# ============================================================

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
        "assists",
        "rebounds",
        "fga",
        "three_pa"
    ]

    for position in (
        "G",
        "F",
        "C",
        "OTHER"
    ):

        players = players_by_position.get(
            position,
            []
        )

        position_result = {

            "players_count": len(players),

            "assists": None,

            "rebounds": None,

            "fga": None,

            "three_pa": None,

            "stat_players_count": {},

            "players": players
        }

        for stat_key in stat_keys:

            values = [
                player[stat_key]
                for player in players
                if player.get(stat_key) is not None
            ]

            position_result[
                "stat_players_count"
            ][stat_key] = len(values)

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


# ============================================================
# DEFESA POR POSIÇÃO COM FILTRO DE MINUTAGEM
# ============================================================

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

            if player_minutes is None:
                continue

            if player_minutes < minutes_threshold:
                continue

            if stat_value is None:
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


# ============================================================
# HISTÓRICO DE PARTIDAS DA EQUIPE
# ============================================================

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


# ============================================================
# CACHE DE RESUMOS DE PARTIDAS ESPN
# ============================================================

@lru_cache(maxsize=500)
def get_game_summary(
    event_id: str
) -> dict:

    return request_json(
        SUMMARY_URL,
        {"event": event_id}
    )


@lru_cache(maxsize=60)
def get_team_defense_history(
    team_abbr,
    limit=10,
    target_date=None
):

    if target_date is None:
        target_date = date.today()

    logger.info(
        "Consultando histórico defensivo: equipe=%s data=%s limite=%s",
        team_abbr,
        target_date,
        limit
    )

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

        for future in as_completed(
            futures
        ):

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
        "rebounds_allowed",
        "assists_allowed"
    ]

    position_stat_keys = [
        "assists",
        "rebounds",
        "fga",
        "three_pa"
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

        for position in (
            "G",
            "F",
            "C",
            "OTHER"
        ):

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

                        game_values.append(
                            value
                        )

                        individual_total += value

                    player_count = (
                        position_data
                        .get("stat_players_count", {})
                        .get(stat_key, 0)
                    )

                    individual_appearances += (
                        player_count
                    )

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
                game.get(
                    "positions",
                    {}
                ).get(
                    position,
                    {}
                ).get(
                    "players_count",
                    0
                )
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

        "averages": calculate_averages(
            results
        ),

        "positions": calculate_position_averages(
            results
        )
    }

    for window in (
        5,
        10
    ):

        sample = results[:window]

        history[f"L{window}"] = {

            "team": team_abbr,

            "games_analyzed": len(sample),

            "games": sample,

            "averages": calculate_averages(
                sample
            ),

            "positions": calculate_position_averages(
                sample
            )
        }

    logger.info(
        "Histórico defensivo concluído: equipe=%s jogos=%s",
        team_abbr,
        len(results)
    )

    return history


# ============================================================
# AGENDA NBA
# ============================================================

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

    for event in data.get(
        "events",
        []
    ):

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

            if competitor.get(
                "homeAway"
            ) == "home":

                home = item

            elif competitor.get(
                "homeAway"
            ) == "away":

                away = item

        if home and away:

            games.append({

                "game_id": event.get("id"),

                "home": home,

                "away": away,

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


# ============================================================
# ELENCOS DOS TIMES
# ============================================================

def collect_roster_players(
    node,
    players
):

    if isinstance(
        node,
        dict
    ):

        athlete_id = node.get("id")

        name = node.get("fullName")

        if athlete_id and name:

            athlete_id = str(
                athlete_id
            )

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
                existing.get("position")
                == "OTHER"
                and position != "OTHER"
            ):

                existing["position"] = position

        for value in node.values():

            collect_roster_players(
                value,
                players
            )

    elif isinstance(
        node,
        list
    ):

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

    collect_roster_players(
        data,
        players
    )

    return tuple(
        players.values()
    )


# ============================================================
# JOGADORES DAS EQUIPES DA DATA SELECIONADA
# ============================================================

def get_players_for_date(
    selected_date: str
) -> dict:

    schedule = fetch_espn_games(
        selected_date
    )

    teams = set()

    for game in schedule["games"]:

        teams.add(
            game["home"]["abbr"]
        )

        teams.add(
            game["away"]["abbr"]
        )

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

        for future in as_completed(
            futures
        ):

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

                    entry["teams"].add(
                        team
                    )

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
                        ] = player[
                            "position"
                        ]

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

            logger.warning(
                "Jogador com associação ambígua: %s - equipes: %s",
                entry["player"]["player"],
                entry["teams"]
            )

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


# ============================================================
# HISTÓRICO INDIVIDUAL DOS JOGADORES
# ============================================================

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

    event_dates = data.get(
        "events",
        {}
    )

    games = {}

    for month in season_category.get(
        "categories",
        []
    ):

        for event in month.get(
            "events",
            []
        ):

            event_id = str(
                event.get("eventId", "")
            )

            stats = event.get(
                "stats",
                []
            )

            if (
                not event_id
                or len(stats) <= 13
            ):
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

                "fga": parse_attempts(
                    stats[1]
                ),

                "three_pa": parse_attempts(
                    stats[3]
                ),

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


# ============================================================
# NOVO:
# EXTRAÇÃO DE RESULTADO INDIVIDUAL DIRETAMENTE DO SUMMARY
# ============================================================

def extract_player_game_from_summary(
    summary,
    player_id
):

    """
    Localiza um jogador dentro do boxscore da partida
    pelo ID ESPN e retorna suas estatísticas.

    Essa função é usada na validação pós-jogo porque
    o Summary pertence diretamente à partida encerrada.
    """

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

        team_data = team_block.get(
            "team",
            {}
        )

        team_abbr = (
            team_data.get("abbreviation")
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
                )

                if raw_stats is None:

                    raw_stats = athlete_entry.get(
                        "statistics"
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

                return {

                    "player_id": player_id,

                    "player": player_name,

                    "team": team_abbr,

                    "minutes": minutes,

                    "assists": assists,

                    "rebounds": rebounds,

                    "points": points,

                    "fga": fga,

                    "three_pa": three_pa
                }

    return None


# ============================================================
# PARTICIPAÇÃO DO JOGADOR NOS ÚLTIMOS JOGOS DA EQUIPE
# ============================================================

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

    for window in (
        5,
        10
    ):

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


# ============================================================
# IDENTIFICA ADVERSÁRIOS DA DATA
# ============================================================

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


# ============================================================
# CONSTRUÇÃO DO RESULTADO INDIVIDUAL
# ============================================================

def build_player_result(
    player,
    games,
    target_date,
    market,
    defense=None,
    opponent=None
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
        "attempts": "fga"
    }[market]

    defense_key = {
        "points": "fga_allowed",
        "assists": "assists_allowed",
        "rebounds": "rebounds_allowed",
        "threes": "three_pa_allowed",
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
        "assists": "assists",
        "rebounds": "rebounds",
        "threes": "three_pa",
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

    if (
        defense
        and position in (
            "G",
            "F",
            "C"
        )
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

    team_game_counts = get_player_team_game_counts(
        games,
        player["team"],
        target_date,
        season_year(target_date)
    )

    return {

        "player_id": str(
            player["id"]
        ),

        "player": player["player"],

        "team": player["team"],

        "position": position,

        "opponent": opponent,

        "last_game_team": last5[0].get(
            "team"
        ),

        "team_changed": bool(
            last5[0].get("team")
            and last5[0]["team"] != player["team"]
        ),

        "last5_avg": round(
            avg5,
            2
        ),

        "last10_avg": round(
            avg10,
            2
        ),

        "minutes_last5_avg": round(
            minutes_avg,
            1
        ),

        "triage_score": round(
            score,
            2
        ),

        "games_last5": len(last5),

        "games_last10": len(last10),

        **team_game_counts,

        "opponent_defense_l5_avg":
            defense_l5,

        "opponent_defense_l10_avg":
            defense_l10,

        "opponent_defense_games_l5":
            defense_games_l5,

        "opponent_defense_games_l10":
            defense_games_l10,

        "opponent_position_defense_l5_avg":
            position_defense_l5,

        "opponent_position_defense_l10_avg":
            position_defense_l10,

        "opponent_position_defense_games_l5":
            position_games_l5,

        "opponent_position_defense_games_l10":
            position_games_l10,

        "opponent_position_individual_defense_l5_avg":
            position_individual_l5,

        "opponent_position_individual_defense_l10_avg":
            position_individual_l10,

        "opponent_position_individual_defense_player_games_l5":
            position_player_games_l5,

        "opponent_position_individual_defense_player_games_l10":
            position_player_games_l10,

        "opponent_position_minutes_filtered_threshold":
            round(
                minutes_avg,
                1
            ),

        "opponent_position_minutes_filtered_defense_l5_avg":
            (
                position_filtered_l5.get(
                    "average_per_player_appearance"
                )
                if position_filtered_l5
                else None
            ),

        "opponent_position_minutes_filtered_defense_l10_avg":
            (
                position_filtered_l10.get(
                    "average_per_player_appearance"
                )
                if position_filtered_l10
                else None
            ),

        "opponent_position_minutes_filtered_defense_l5_per_minute":
            (
                position_filtered_l5.get(
                    "per_minute"
                )
                if position_filtered_l5
                else None
            ),

        "opponent_position_minutes_filtered_defense_l10_per_minute":
            (
                position_filtered_l10.get(
                    "per_minute"
                )
                if position_filtered_l10
                else None
            ),

        "opponent_position_minutes_filtered_defense_l5_projected":
            (
                position_filtered_l5.get(
                    "projected_at_player_minutes"
                )
                if position_filtered_l5
                else None
            ),

        "opponent_position_minutes_filtered_defense_l10_projected":
            (
                position_filtered_l10.get(
                    "projected_at_player_minutes"
                )
                if position_filtered_l10
                else None
            ),

        "opponent_position_minutes_filtered_defense_games_l5":
            (
                position_filtered_l5.get(
                    "games_analyzed",
                    0
                )
                if position_filtered_l5
                else 0
            ),

        "opponent_position_minutes_filtered_defense_games_l10":
            (
                position_filtered_l10.get(
                    "games_analyzed",
                    0
                )
                if position_filtered_l10
                else 0
            ),

        "opponent_position_minutes_filtered_defense_player_games_l5":
            (
                position_filtered_l5.get(
                    "player_appearances",
                    0
                )
                if position_filtered_l5
                else 0
            ),

        "opponent_position_minutes_filtered_defense_player_games_l10":
            (
                position_filtered_l10.get(
                    "player_appearances",
                    0
                )
                if position_filtered_l10
                else 0
            ),

        "opponent_position_minutes_filtered_defense_minutes_l5":
            (
                position_filtered_l5.get(
                    "total_minutes",
                    0
                )
                if position_filtered_l5
                else 0
            ),

        "opponent_position_minutes_filtered_defense_minutes_l10":
            (
                position_filtered_l10.get(
                    "total_minutes",
                    0
                )
                if position_filtered_l10
                else 0
            )
    }


# ============================================================
# COLETA DOS CANDIDATOS
# ============================================================

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

    year = season_year(
        target_date
    )

    stage_start = time.perf_counter()

    roster_data = get_players_for_date(
        selected_date
    )

    players = roster_data["players"]

    opponents = get_opponents_for_date(
        roster_data["schedule"]
    )

    logger.info(
        "Etapa 1 concluída: %s equipes e %s jogadores. "
        "Tempo: %.2fs",
        roster_data["teams_found"],
        len(players),
        time.perf_counter() - stage_start
    )

    stage_start = time.perf_counter()

    defense_cache = {}

    unique_opponents = set(
        opponents.values()
    )

    def fetch_team_defense(team):

        defense = get_team_defense_history(
            team,
            10,
            target_date
        )

        return team, defense

    with ThreadPoolExecutor(
        max_workers=3
    ) as executor:

        futures = [
            executor.submit(
                fetch_team_defense,
                team
            )
            for team in unique_opponents
        ]

        for future in as_completed(
            futures
        ):

            try:

                team, defense = future.result()

                defense_cache[team] = defense

            except Exception as error:

                logger.warning(
                    "Falha ao consultar histórico defensivo: %s",
                    error
                )

    logger.info(
        "Etapa 2 concluída: histórico defensivo de %s/%s equipes. "
        "Tempo: %.2fs",
        len(defense_cache),
        len(unique_opponents),
        time.perf_counter() - stage_start
    )

    stage_start = time.perf_counter()

    results = []

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

        for future in as_completed(
            futures
        ):

            try:

                player, logs = future.result()

                successful_logs += 1

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
                    opponent
                )

                if result:
                    results.append(result)

            except Exception as error:

                errors += 1

                logger.warning(
                    "Falha ao consultar jogador: %s",
                    error
                )

    logger.info(
        "Etapa 3 concluída: %s históricos consultados, "
        "%s erros e %s candidatos. Tempo: %.2fs",
        successful_logs,
        errors,
        len(results),
        time.perf_counter() - stage_start
    )

    results.sort(
        key=lambda item: item["triage_score"],
        reverse=True
    )

    save_candidate_snapshots(
        selected_date,
        market,
        results
    )

    final_limit = min(
        limit,
        10
    )

    logger.info(
        "Consulta finalizada: data=%s mercado=%s "
        "candidatos=%s total=%.2fs",
        selected_date,
        market,
        final_limit,
        time.perf_counter() - total_start
    )

    return {

        "date": selected_date,

        "market": market,

        "market_label":
            MARKETS[market]["label"],

        "count": min(
            len(results),
            final_limit
        ),

        "source": "ESPN",

        "season": season_for(
            target_date
        ),

        "teams_found":
            roster_data["teams_found"],

        "players_found":
            len(players),

        "players_consulted":
            successful_logs,

        "players_with_errors":
            errors,

        "updated_at":
            datetime.now().isoformat(),

        "method_note": (
            "Triagem estatística baseada exclusivamente na temporada "
            "regular atual da NBA. Médias individuais dos últimos 5 "
            "e 10 jogos anteriores à data selecionada. Também informa "
            "quantos dos últimos 5 e 10 jogos concluídos da equipe atual "
            "o jogador disputou; esse indicador é informativo e não altera "
            "a pontuação. Filtro mínimo de 24 minutos médios nos últimos "
            "cinco jogos. Inclui indicadores defensivos gerais do "
            "adversário e indicadores defensivos por posição nos períodos "
            "L5 e L10. Os indicadores por posição apresentam a produção "
            "total média por jogo e a média individual por participação "
            "de jogador. Também incluem uma análise adicional filtrada "
            "pela minutagem: considera somente jogadores da mesma posição "
            "que tenham atuado minutos iguais ou superiores à média de "
            "minutos do candidato nos últimos cinco jogos. Para esse grupo, "
            "calcula a média por participação, a taxa por minuto e uma "
            "projeção ajustada à minutagem do candidato. A amostra filtrada "
            "informa partidas, participações e minutos considerados. "
            "Esses indicadores são contexto complementar e não fazem parte "
            "da pontuação. Não considera linhas, odds, lesões ou escalações. "
            "A pontuação não representa probabilidade de acerto."
        ),

        "candidates":
            results[:final_limit]
    }


# ============================================================
# ENDPOINTS PRINCIPAIS
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "service": "NBA Props Scanner",

        "version": "3.13.0",

        "source": "ESPN"
    }


@app.get("/health/db")
def database_health():

    if not DATABASE_URL:

        return {

            "status": "not_configured",

            "database": "Neon PostgreSQL",

            "history_enabled": False
        }

    try:

        with get_database_connection() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SELECT COUNT(*) FROM candidate_snapshots"
                )

                total_snapshots = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM candidate_results"
                )

                total_results = cursor.fetchone()[0]

        return {

            "status": "connected",

            "database": "Neon PostgreSQL",

            "history_enabled": True,

            "total_snapshots":
                total_snapshots,

            "total_results":
                total_results
        }

    except Exception as error:

        logger.exception(
            "Verificação do banco falhou: %s",
            error
        )

        return {

            "status": "error",

            "database": "Neon PostgreSQL",

            "history_enabled": False,

            "message":
                "Não foi possível consultar o banco de dados."
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

    return fetch_espn_games(
        selected_date
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

    if market not in MARKETS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Mercado inválido: "
                + ", ".join(
                    MARKETS.keys()
                )
            )
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

    logger.info(
        "Candidatos: data=%s mercado=%s limite=%s",
        selected_date,
        market,
        limit
    )

    try:

        return collect_candidates(
            selected_date,
            market,
            limit
        )

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Erro ao gerar candidatos"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                f"Não foi possível gerar candidatos: {error}"
            )
        )


# ============================================================
# COLETA DOS RESULTADOS REAIS
# ============================================================

@app.get("/collect-results")
def collect_results(
    selected_date: str = Query(
        alias="date"
    ),

    market: str = Query(
        default="assists"
    )
):

    """
    Coleta resultados reais de partidas finalizadas.

    IMPORTANTE:
    - candidate_snapshots é a fotografia pré-jogo.
    - candidate_results guarda o resultado pós-jogo.
    - O jogador é localizado pelo ESPN player_id diretamente
      no Summary da partida.
    - Ausência de registro não é convertida em zero.
    """

    if market != "assists":

        raise HTTPException(
            status_code=400,
            detail=(
                "Nesta primeira etapa, a coleta está habilitada "
                "apenas para assists."
            )
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

    for game in schedule.get(
        "games",
        []
    ):

        if not game.get("completed"):
            continue

        home = (
            game.get("home", {})
            .get("abbr")
        )

        away = (
            game.get("away", {})
            .get("abbr")
        )

        game_id = str(
            game.get("game_id") or ""
        )

        if (
            home
            and away
            and game_id
        ):

            completed_games[
                (
                    home.upper(),
                    away.upper()
                )
            ] = game_id

            completed_games[
                (
                    away.upper(),
                    home.upper()
                )
            ] = game_id

            completed_game_ids.append(
                game_id
            )

    completed_count = len(
        completed_game_ids
    )

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
                (
                    target_date,
                    market
                )
            )

            snapshots = cursor.fetchall()

    if not snapshots:

        return {

            "date": selected_date,

            "market": market,

            "status": "no_snapshots",

            "snapshots_found": 0,

            "completed_games":
                completed_count,

            "results_saved": 0,

            "pending": 0,

            "message": (
                "Não há candidatos históricos para essa data "
                "e mercado. Consulte os candidatos antes do jogo "
                "para registrar a fotografia."
            )
        }

    summary_cache = {}

    summary_errors = 0

    def fetch_summary(game_id):

        return (
            game_id,
            get_game_summary(
                game_id
            )
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

        for future in as_completed(
            futures
        ):

            try:

                game_id, summary = (
                    future.result()
                )

                summary_cache[
                    game_id
                ] = summary

            except Exception as error:

                summary_errors += 1

                logger.warning(
                    "Falha ao carregar summary do jogo %s: %s",
                    game_id,
                    error
                )

    logger.info(
        "Summaries carregados: %s/%s",
        len(summary_cache),
        len(completed_game_ids)
    )

    results_to_save = []

    pending = 0

    errors = 0

    def fetch_player_result(snapshot):

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

            return (
                "pending",
                None
            )

        summary = summary_cache.get(
            game_id
        )

        if not summary:

            return (
                "pending",
                None
            )

        player_game = (
            extract_player_game_from_summary(
                summary,
                player_id
            )
        )

        if not player_game:

            logger.warning(
                "Jogador não localizado no summary: "
                "%s (%s) jogo=%s",
                player_name,
                player_id,
                game_id
            )

            return (
                "pending",
                None
            )

        minutes = player_game.get(
            "minutes"
        )

        if (
            minutes is None
            or minutes <= 0
        ):

            return (
                "pending",
                None
            )

        actual = player_game.get(
            "assists"
        )

        if actual is None:

            return (
                "pending",
                None
            )

        row = (
            target_date,
            market,
            str(player_id),
            player_name,
            team,
            opponent,
            float(actual),
            game_id,
            Jsonb({

                "minutes":
                    minutes,

                "assists":
                    actual,

                "game_date":
                    selected_date,

                "team_from_summary":
                    player_game.get(
                        "team"
                    ),

                "player_id_from_summary":
                    player_game.get(
                        "player_id"
                    ),

                "source":
                    "ESPN game summary"
            })
        )

        return (
            "ready",
            row
        )

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = [
            executor.submit(
                fetch_player_result,
                row
            )
            for row in snapshots
        ]

        for future in as_completed(
            futures
        ):

            try:

                status, row = (
                    future.result()
                )

                if status == "ready":

                    results_to_save.append(
                        row
                    )

                else:

                    pending += 1

            except Exception as error:

                errors += 1

                logger.warning(
                    "Falha ao coletar resultado individual: %s",
                    error
                )

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

    with get_database_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM candidate_results
                WHERE
                    selected_date = %s
                    AND market = %s
                """,
                (
                    target_date,
                    market
                )
            )

            total_saved = (
                cursor.fetchone()[0]
            )

    return {

        "date": selected_date,

        "market": market,

        "status": "processed",

        "snapshots_found":
            len(snapshots),

        "completed_games":
            completed_count,

        "summaries_loaded":
            len(summary_cache),

        "summary_errors":
            summary_errors,

        "results_detected":
            len(results_to_save),

        "results_saved_this_run":
            inserted,

        "total_results_saved":
            total_saved,

        "pending":
            pending,

        "errors":
            errors,

        "note": (
            "Somente jogos finalizados e jogadores "
            "individualmente localizados no Summary ESPN "
            "são gravados. Ausência de registro não é "
            "convertida em zero; candidatos pendentes "
            "podem ser consultados novamente."
        )
    }


# ============================================================
# NOVO ENDPOINT:
# VALIDAÇÃO PRÉ-JOGO X RESULTADO REAL
# ============================================================

@app.get("/validation")
def validation(
    selected_date: str = Query(
        alias="date"
    ),

    market: str = Query(
        default="assists"
    )
):

    """
    Cruza a fotografia pré-jogo armazenada em
    candidate_snapshots com o resultado real armazenado
    em candidate_results.

    Este endpoint NÃO calcula green/red.

    Ele mede a capacidade da triagem em relação ao
    resultado real da partida.
    """

    if market not in MARKETS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Mercado inválido: "
                + ", ".join(
                    MARKETS.keys()
                )
            )
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

    if not DATABASE_URL:

        raise HTTPException(
            status_code=503,
            detail="Banco de dados não configurado."
        )

    # --------------------------------------------------------
    # BUSCA SNAPSHOTS E RESULTADOS
    # --------------------------------------------------------

    with get_database_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    player_id,
                    player_name,
                    team,
                    opponent,
                    position,
                    triage_score,
                    last5_avg,
                    last10_avg,
                    minutes_last5_avg,
                    payload,
                    captured_at
                FROM candidate_snapshots
                WHERE
                    selected_date = %s
                    AND market = %s
                ORDER BY triage_score DESC NULLS LAST
                """,
                (
                    target_date,
                    market
                )
            )

            snapshots = cursor.fetchall()

            cursor.execute(
                """
                SELECT
                    player_id,
                    player_name,
                    team,
                    opponent,
                    actual_value,
                    game_id,
                    collected_at,
                    payload
                FROM candidate_results
                WHERE
                    selected_date = %s
                    AND market = %s
                """,
                (
                    target_date,
                    market
                )
            )

            result_rows = cursor.fetchall()

    # --------------------------------------------------------
    # INDEXA RESULTADOS PELO PLAYER_ID
    # --------------------------------------------------------

    results_by_player = {}

    for row in result_rows:

        (
            player_id,
            player_name,
            team,
            opponent,
            actual_value,
            game_id,
            collected_at,
            payload
        ) = row

        results_by_player[
            str(player_id)
        ] = {

            "player_name":
                player_name,

            "team":
                team,

            "opponent":
                opponent,

            "actual_value":
                float(actual_value)
                if actual_value is not None
                else None,

            "game_id":
                game_id,

            "collected_at":
                collected_at.isoformat()
                if collected_at
                else None,

            "payload":
                payload or {}
        }

    # --------------------------------------------------------
    # MONTA A VALIDAÇÃO INDIVIDUAL
    # --------------------------------------------------------

    items = []

    completed_count = 0

    pending_count = 0

    actual_values = []

    l5_errors = []

    l10_errors = []

    score_values = []

    score_actual_pairs = []

    above_l5 = 0

    below_l5 = 0

    equal_l5 = 0

    above_l10 = 0

    below_l10 = 0

    equal_l10 = 0

    for snapshot in snapshots:

        (
            snapshot_id,
            player_id,
            player_name,
            team,
            opponent,
            position,
            triage_score,
            last5_avg,
            last10_avg,
            minutes_last5_avg,
            payload,
            captured_at
        ) = snapshot

        player_id = str(
            player_id
        )

        result = results_by_player.get(
            player_id
        )

        snapshot_payload = (
            payload or {}
        )

        if result:

            actual = result["actual_value"]

            status = "completed"

            diff_l5 = None
            diff_l10 = None

            pct_l5 = None
            pct_l10 = None

            if last5_avg is not None:

                diff_l5 = round(
                    actual - float(last5_avg),
                    2
                )

                if float(last5_avg) != 0:

                    pct_l5 = round(
                        (
                            diff_l5
                            / float(last5_avg)
                        ) * 100,
                        2
                    )

                if actual > float(last5_avg):

                    above_l5 += 1

                    relation_l5 = "above"

                elif actual < float(last5_avg):

                    below_l5 += 1

                    relation_l5 = "below"

                else:

                    equal_l5 += 1

                    relation_l5 = "equal"

            else:

                relation_l5 = None

            if last10_avg is not None:

                diff_l10 = round(
                    actual - float(last10_avg),
                    2
                )

                if float(last10_avg) != 0:

                    pct_l10 = round(
                        (
                            diff_l10
                            / float(last10_avg)
                        ) * 100,
                        2
                    )

                if actual > float(last10_avg):

                    above_l10 += 1

                    relation_l10 = "above"

                elif actual < float(last10_avg):

                    below_l10 += 1

                    relation_l10 = "below"

                else:

                    equal_l10 += 1

                    relation_l10 = "equal"

            else:

                relation_l10 = None

            if last5_avg is not None:

                l5_errors.append(
                    abs(
                        actual
                        - float(last5_avg)
                    )
                )

            if last10_avg is not None:

                l10_errors.append(
                    abs(
                        actual
                        - float(last10_avg)
                    )
                )

            actual_values.append(
                actual
            )

            if triage_score is not None:

                score_values.append(
                    float(triage_score)
                )

                score_actual_pairs.append(
                    (
                        float(triage_score),
                        actual
                    )
                )

            completed_count += 1

        else:

            status = "pending"

            pending_count += 1

            actual = None

            diff_l5 = None
            diff_l10 = None

            pct_l5 = None
            pct_l10 = None

            relation_l5 = None
            relation_l10 = None

        items.append({

            "player_id":
                player_id,

            "player":
                player_name,

            "team":
                team,

            "opponent":
                opponent,

            "position":
                position,

            "triage_score":
                (
                    float(triage_score)
                    if triage_score is not None
                    else None
                ),

            "last5_avg":
                (
                    float(last5_avg)
                    if last5_avg is not None
                    else None
                ),

            "last10_avg":
                (
                    float(last10_avg)
                    if last10_avg is not None
                    else None
                ),

            "minutes_last5_avg":
                (
                    float(minutes_last5_avg)
                    if minutes_last5_avg is not None
                    else None
                ),

            "actual_value":
                actual,

            "difference_vs_l5":
                diff_l5,

            "difference_vs_l10":
                diff_l10,

            "difference_pct_vs_l5":
                pct_l5,

            "difference_pct_vs_l10":
                pct_l10,

            "relation_vs_l5":
                relation_l5,

            "relation_vs_l10":
                relation_l10,

            "status":
                status,

            "game_id":
                (
                    result["game_id"]
                    if result
                    else None
                ),

            "result_payload":
                (
                    result["payload"]
                    if result
                    else None
                ),

            "snapshot_payload":
                snapshot_payload,

            "captured_at":
                (
                    captured_at.isoformat()
                    if captured_at
                    else None
                )
        })

    # --------------------------------------------------------
    # ESTATÍSTICAS GERAIS
    # --------------------------------------------------------

    def safe_average(values):

        if not values:
            return None

        return round(
            sum(values) / len(values),
            2
        )

    def safe_median(values):

        if not values:
            return None

        ordered = sorted(values)

        size = len(ordered)

        middle = size // 2

        if size % 2:

            return round(
                ordered[middle],
                2
            )

        return round(
            (
                ordered[middle - 1]
                + ordered[middle]
            ) / 2,
            2
        )

    def calculate_pearson(pairs):

        if len(pairs) < 2:
            return None

        xs = [
            pair[0]
            for pair in pairs
        ]

        ys = [
            pair[1]
            for pair in pairs
        ]

        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)

        numerator = sum(
            (
                x - x_mean
            )
            * (
                y - y_mean
            )
            for x, y in pairs
        )

        denominator_x = sum(
            (
                x - x_mean
            ) ** 2
            for x in xs
        )

        denominator_y = sum(
            (
                y - y_mean
            ) ** 2
            for y in ys
        )

        denominator = (
            denominator_x
            * denominator_y
        ) ** 0.5

        if denominator == 0:
            return None

        return round(
            numerator / denominator,
            4
        )

    mae_l5 = safe_average(
        l5_errors
    )

    mae_l10 = safe_average(
        l10_errors
    )

    summary = {

        "snapshots_found":
            len(snapshots),

        "results_found":
            len(result_rows),

        "completed":
            completed_count,

        "pending":
            pending_count,

        "completion_rate_pct":
            (
                round(
                    completed_count
                    / len(snapshots)
                    * 100,
                    2
                )
                if snapshots
                else 0
            ),

        "actual_average":
            safe_average(
                actual_values
            ),

        "actual_median":
            safe_median(
                actual_values
            ),

        "l5_mean_absolute_error":
            mae_l5,

        "l10_mean_absolute_error":
            mae_l10,

        "l5_above_count":
            above_l5,

        "l5_below_count":
            below_l5,

        "l5_equal_count":
            equal_l5,

        "l10_above_count":
            above_l10,

        "l10_below_count":
            below_l10,

        "l10_equal_count":
            equal_l10,

        "score_actual_pearson":
            calculate_pearson(
                score_actual_pairs
            )
    }

    # --------------------------------------------------------
    # RETORNO
    # --------------------------------------------------------

    return {

        "date":
            selected_date,

        "market":
            market,

        "market_label":
            MARKETS[market]["label"],

        "status":
            "ok",

        "summary":
            summary,

        "items":
            items,

        "method_note": (
            "Esta validação cruza a fotografia pré-jogo "
            "preservada em candidate_snapshots com o resultado "
            "real armazenado em candidate_results. "
            "O endpoint não utiliza linhas de casas de apostas "
            "e não classifica green/red. Diferenças positivas "
            "indicam que o jogador produziu acima da média "
            "histórica usada na análise; diferenças negativas "
            "indicam produção abaixo dessa média. "
            "O campo score_actual_pearson representa apenas "
            "a correlação linear entre triage_score e resultado "
            "real dentro da amostra disponível."
        )
    }


# ============================================================
# ENDPOINT DE DIAGNÓSTICO DO GAMELOG
# ============================================================

@app.get("/debug-gamelog")
def debug_gamelog(
    player_id: str = Query(
        ...,
        description="ESPN athlete/player ID"
    ),

    year: Optional[int] = Query(
        default=None
    )
):

    """
    Diagnóstico real do gamelog de um jogador.

    Antes este endpoint estava fixado em athlete_id=1966.
    Agora ele utiliza o player_id informado na URL.
    """

    if not player_id:

        raise HTTPException(
            status_code=400,
            detail="player_id é obrigatório."
        )

    if year is None:

        year = season_year(
            date.today()
        )

    data = request_json(
        GAMELOG_URL.format(
            athlete=str(
                player_id
            ).strip()
        ),
        {
            "season": year
        }
    )

    events = data.get(
        "events"
    )

    if isinstance(
        events,
        dict
    ):

        events_sample = [
            {
                "event_id": key,
                "data": value
            }
            for key, value in list(
                events.items()
            )[:3]
        ]

        events_count = len(
            events
        )

    elif isinstance(
        events,
        list
    ):

        events_sample = events[:3]

        events_count = len(
            events
        )

    else:

        events_sample = events

        events_count = 0

    return {

        "athlete_id":
            str(player_id).strip(),

        "season_year":
            year,

        "top_level_keys":
            list(data.keys()),

        "categories_present":
            "categories" in data,

        "labels":
            data.get("labels"),

        "names":
            data.get("names"),

        "season_types":
            data.get("seasonTypes"),

        "events_type":
            type(events).__name__,

        "events_count":
            events_count,

        "events_sample":
            events_sample
    }


# ============================================================
# STATS CHECK
# ============================================================

@app.get("/stats-check")
def stats_check(
    selected_date: str = Query(
        alias="date"
    )
):

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

    data = get_players_for_date(
        selected_date
    )

    players = data["players"]

    if not players:

        raise HTTPException(
            status_code=503,
            detail=(
                "Nenhum jogador encontrado "
                "nos elencos consultados."
            )
        )

    year = season_year(
        target_date
    )

    sample_player = None

    previous_games = []

    players_checked = 0

    players_with_errors = 0

    for player in players[:60]:

        players_checked += 1

        try:

            sample_logs = get_athlete_logs(
                player["id"],
                year
            )

            player_games = [
                game
                for game in sample_logs
                if game["date"] < selected_date
            ]

            if player_games:

                sample_player = player

                previous_games = player_games

                break

        except Exception:

            players_with_errors += 1

            continue

    return {

        "status": (
            "ok"
            if sample_player
            else "warning"
        ),

        "source": "ESPN",

        "season": season_for(
            target_date
        ),

        "season_year":
            year,

        "teams_found":
            data["teams_found"],

        "players_found":
            len(players),

        "players_checked":
            players_checked,

        "players_with_errors":
            players_with_errors,

        "sample_player":
            (
                sample_player["player"]
                if sample_player
                else None
            ),

        "sample_player_id":
            (
                sample_player["id"]
                if sample_player
                else None
            ),

        "sample_games":
            len(previous_games),

        "sample_game":
            (
                previous_games[0]
                if previous_games
                else None
            ),

        "checked_at":
            datetime.now().isoformat()
    }


# ============================================================
# TEAM CHECK
# ============================================================

@app.get("/team-check")
def team_check(
    selected_date: str = Query(
        alias="date"
    )
):

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

    data = get_players_for_date(
        selected_date
    )

    players = data["players"]

    if not players:

        raise HTTPException(
            status_code=503,
            detail=(
                "Nenhum jogador encontrado "
                "nos elencos consultados."
            )
        )

    year = season_year(
        target_date
    )

    def check_player(player):

        logs = get_athlete_logs(
            player["id"],
            year
        )

        previous_games = [
            game
            for game in logs
            if game["date"] < selected_date
        ]

        if not previous_games:

            return {

                "player":
                    player["player"],

                "player_id":
                    player["id"],

                "current_team":
                    player["team"],

                "status":
                    "no_previous_games"
            }

        last_game = previous_games[0]

        historical_teams = sorted({
            game.get("team")
            for game in previous_games[:5]
            if game.get("team")
        })

        last_game_team = (
            last_game.get("team")
        )

        if not last_game_team:

            status = (
                "missing_historical_team"
            )

        elif (
            last_game_team
            != player["team"]
        ):

            status = (
                "team_mismatch"
            )

        else:

            status = "ok"

        return {

            "player":
                player["player"],

            "player_id":
                player["id"],

            "current_team":
                player["team"],

            "last_game_date":
                last_game["date"],

            "last_game_team":
                last_game_team,

            "team_changed":
                bool(
                    last_game_team
                    and last_game_team
                    != player["team"]
                ),

            "teams_in_last5":
                historical_teams,

            "multiple_teams_last5":
                len(historical_teams) > 1,

            "status":
                status
        }

    checked_results = []

    errors = []

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = {
            executor.submit(
                check_player,
                player
            ): player
            for player in players
        }

        for future in as_completed(
            futures
        ):

            player = futures[future]

            try:

                checked_results.append(
                    future.result()
                )

            except Exception as error:

                errors.append({

                    "player":
                        player["player"],

                    "player_id":
                        player["id"],

                    "current_team":
                        player["team"],

                    "error":
                        str(error)[:250]
                })

    checked_results.sort(
        key=lambda item:
            item["player"].casefold()
    )

    errors.sort(
        key=lambda item:
            item["player"].casefold()
    )

    mismatches = [
        item
        for item in checked_results
        if item["status"]
        == "team_mismatch"
    ]

    missing_historical_team = [
        item
        for item in checked_results
        if item["status"]
        == "missing_historical_team"
    ]

    no_previous_games = [
        item
        for item in checked_results
        if item["status"]
        == "no_previous_games"
    ]

    multiple_teams = [
        item
        for item in checked_results
        if item.get(
            "multiple_teams_last5"
        )
    ]

    return {

        "status":
            (
                "ok"
                if not errors
                else "partial"
            ),

        "source":
            "ESPN",

        "date":
            selected_date,

        "season":
            season_for(
                target_date
            ),

        "teams_found":
            data["teams_found"],

        "players_found":
            len(players),

        "players_excluded_multiple_rosters":
            data.get(
                "players_excluded_multiple_teams",
                0
            ),

        "players_checked":
            (
                len(checked_results)
                + len(errors)
            ),

        "players_with_history":
            (
                len(checked_results)
                - len(no_previous_games)
            ),

        "players_without_history":
            len(no_previous_games),

        "players_with_team_mismatch":
            len(mismatches),

        "players_missing_historical_team":
            len(
                missing_historical_team
            ),

        "players_with_multiple_teams_last5":
            len(multiple_teams),

        "players_with_errors":
            len(errors),

        "team_mismatches":
            mismatches,

        "missing_historical_team":
            missing_historical_team,

        "multiple_teams_last5":
            multiple_teams,

        "players_without_history":
            no_previous_games,

        "errors":
            errors,

        "checked_at":
            datetime.now().isoformat()
    }


# ============================================================
# DEFESA CHECK
# ============================================================

@app.get("/defense-check")
def defense_check():

    summary = request_json(
        SUMMARY_URL,
        {"event": "0401909918"}
    )

    return {

        "POR_defense":
            extract_team_defense_stats(
                summary,
                "POR"
            ),

        "DEN_defense":
            extract_team_defense_stats(
                summary,
                "DEN"
            )
    }


# ============================================================
# DEFESA POR POSIÇÃO
# ============================================================

@app.get("/defense-position-check")
def defense_position_check(
    event: str = Query(...),
    defending_team: str = Query(...)
):

    defending_team = (
        defending_team.upper()
    )

    summary = request_json(
        SUMMARY_URL,
        {"event": event}
    )

    data = extract_opponent_players_by_position(
        summary,
        defending_team
    )

    return {

        "event":
            event,

        "defending_team":
            defending_team,

        "opponent":
            data["opponent"],

        "positions":
            data["players"]
    }


# ============================================================
# DIAGNÓSTICO BRUTO DA ESTRUTURA ESPN
# ============================================================

@app.get("/defense-position-raw-check")
def defense_position_raw_check(
    event: str = Query(...),
    defending_team: str = Query(...)
):

    defending_team = (
        defending_team.upper()
    )

    summary = request_json(
        SUMMARY_URL,
        {"event": event}
    )

    teams = (
        summary
        .get("boxscore", {})
        .get("teams", [])
    )

    players = (
        summary
        .get("boxscore", {})
        .get("players", [])
    )

    return {

        "event":
            event,

        "defending_team":
            defending_team,

        "teams_count":
            len(teams),

        "players_count":
            len(players),

        "players_raw":
            players
    }


# ============================================================
# SCHEDULE CHECK
# ============================================================

@app.get("/schedule-check")
def schedule_check():

    target_date = date.today()

    data = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/teams/den/schedule",
        {
            "season":
                season_year(target_date)
        }
    )

    events = data.get(
        "events",
        []
    )

    return {

        "top_level_keys":
            list(data.keys()),

        "team":
            data.get("team", {}),

        "events_count":
            len(events),

        "sample_events":
            events[:2]
    }


# ============================================================
# DEFESA HISTORY CHECK
# ============================================================

@app.get("/defense-history-check")
def defense_history_check():

    target_date = date.today()

    schedule = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/teams/den/schedule",
        {
            "season":
                season_year(target_date)
        }
    )

    return {

        "team":
            "DEN",

        "games":
            get_completed_team_games(
                schedule,
                "DEN",
                5
            )
    }


# ============================================================
# DEFESA GAME CHECK
# ============================================================

@app.get("/defense-game-check")
def defense_game_check():

    summary = request_json(
        SUMMARY_URL,
        {"event": "401909918"}
    )

    return {

        "DEN_defense":
            extract_team_defense_stats(
                summary,
                "DEN"
            ),

        "POR_defense":
            extract_team_defense_stats(
                summary,
                "POR"
            )
    }


# ============================================================
# DEFESA AVERAGE CHECK
# ============================================================

@app.get("/defense-average-check")
def defense_average_check(
    selected_date: Optional[str] = Query(
        default=None,
        alias="date"
    )
):

    if selected_date:

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

    else:

        target_date = date.today()

    return get_team_defense_history(
        "DEN",
        10,
        target_date
    )
