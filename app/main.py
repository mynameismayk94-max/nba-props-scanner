import logging
import time

from datetime import datetime, date
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nba_props_scanner")

app = FastAPI(
    title="NBA Props Scanner",
    description="API de triagem estatística NBA via ESPN.",
    version="3.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {"User-Agent": "Mozilla/5.0 NBA Props Scanner"}

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

MARKETS = {
    "points": {"index": 13, "label": "Pontos"},
    "assists": {"index": 8, "label": "Assistências"},
    "rebounds": {"index": 7, "label": "Rebotes"},
    "threes": {"index": 3, "label": "Tentativas de 3 pontos"},
    "attempts": {"index": 1, "label": "Tentativas totais"}
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

    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_year(selected_date: date) -> int:
    return (
        selected_date.year + 1
        if selected_date.month >= 10
        else selected_date.year
    )


# ============================================================
# CONSULTA ESPN
# ============================================================

def request_json(url: str, params=None) -> dict:
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
            return float(minutes) + float(seconds) / 60
        except (ValueError, TypeError):
            return 0.0

    return parse_number(value)


def parse_attempts(value) -> float:
    if not value:
        return 0.0

    try:
        parts = str(value).split("-")

        if len(parts) != 2:
            return 0.0

        return float(parts[1])

    except (ValueError, TypeError):
        return 0.0


# ============================================================
# ESTATÍSTICAS DEFENSIVAS DE UMA PARTIDA
# ============================================================

def extract_team_defense_stats(summary, defending_team):

    teams = (
        summary.get("boxscore", {})
        .get("teams", [])
    )

    opponent_stats = None

    for item in teams:

        team_abbr = (
            item.get("team", {})
            .get("abbreviation")
        )

        if team_abbr != defending_team:
            opponent_stats = item.get("statistics", [])
            break

    if opponent_stats is None:
        return None

    stats = {
        stat.get("name"): stat.get("displayValue")
        for stat in opponent_stats
    }

    return {
        "fga_allowed": parse_attempts(
            stats.get("fieldGoalsMade-fieldGoalsAttempted")
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
# NOVO:
# JOGADORES DO ADVERSÁRIO AGRUPADOS POR POSIÇÃO
# ============================================================

def extract_opponent_players_by_position(
    summary,
    defending_team
):
    """
    Extrai os jogadores do adversário agrupados por posição.

    Grupos utilizados:
        G      = Guard
        F      = Forward
        C      = Center
        OTHER  = posição não identificada

    Retorna também as estatísticas individuais:
        assists
        rebounds
        fga
        three_pa
    """

    result = {
        "G": [],
        "F": [],
        "C": [],
        "OTHER": []
    }

    boxscore = summary.get("boxscore", {})

    teams = boxscore.get("teams", [])
    players = boxscore.get("players", [])

    # --------------------------------------------------------
    # Identifica o adversário
    # --------------------------------------------------------

    opponent_team = None

    for team in teams:

        team_abbr = (
            team.get("team", {})
            .get("abbreviation")
        )

        if (
            team_abbr
            and team_abbr.upper() != defending_team.upper()
        ):
            opponent_team = team_abbr.upper()
            break

    if not opponent_team:

        return {
            "opponent": None,
            "players": result
        }

    # --------------------------------------------------------
    # Procura o bloco de jogadores do adversário
    # --------------------------------------------------------

    opponent_players_block = None

    for team_players in players:

        team_abbr = (
            team_players.get("team", {})
            .get("abbreviation")
        )

        if (
            team_abbr
            and team_abbr.upper() == opponent_team
        ):
            opponent_players_block = team_players
            break

    if not opponent_players_block:

        return {
            "opponent": opponent_team,
            "players": result
        }

    # --------------------------------------------------------
    # Processa os jogadores
    # --------------------------------------------------------

    for player_entry in opponent_players_block.get(
        "statistics",
        []
    ):

        athlete = player_entry.get(
            "athlete",
            {}
        )

        player_name = (
            athlete.get("displayName")
            or athlete.get("fullName")
            or athlete.get("shortName")
        )

        if not player_name:
            continue

        # ----------------------------------------------------
        # POSIÇÃO
        # ----------------------------------------------------

        position_data = athlete.get(
            "position",
            {}
        )

        if isinstance(position_data, dict):

            position = (
                position_data.get("abbreviation")
                or position_data.get("displayName")
                or position_data.get("name")
            )

        else:

            position = position_data

        position_upper = str(
            position or ""
        ).strip().upper()

        if position_upper in (
            "G",
            "PG",
            "SG",
            "GUARD"
        ):

            position_group = "G"

        elif position_upper in (
            "F",
            "SF",
            "PF",
            "FORWARD"
        ):

            position_group = "F"

        elif position_upper in (
            "C",
            "CENTER"
        ):

            position_group = "C"

        else:

            position_group = "OTHER"

        # ----------------------------------------------------
        # ESTATÍSTICAS
        # ----------------------------------------------------

        stats = {}

        raw_statistics = player_entry.get(
            "statistics",
            []
        )

        # Em algumas estruturas a ESPN pode fornecer
        # statistics como lista de strings/valores.
        # Quando isso ocorrer, usamos os nomes presentes
        # no bloco do jogador, se disponíveis.

        if isinstance(raw_statistics, list):

            for stat in raw_statistics:

                if isinstance(stat, dict):

                    name = stat.get("name")

                    if name:
                        stats[name] = stat

        def get_stat(*names):

            for name in names:

                if name not in stats:
                    continue

                item = stats[name]

                value = item.get(
                    "value"
                )

                if value is not None:

                    try:
                        return float(value)

                    except (
                        TypeError,
                        ValueError
                    ):
                        pass

                display_value = item.get(
                    "displayValue"
                )

                if display_value is not None:

                    try:
                        return parse_number(
                            display_value
                        )

                    except (
                        TypeError,
                        ValueError
                    ):
                        pass

            return None

        # ----------------------------------------------------
        # Alguns summaries da ESPN usam diretamente os nomes
        # abaixo. Mantemos também aliases para maior robustez.
        # ----------------------------------------------------

        assists = get_stat(
            "assists",
            "AST"
        )

        rebounds = get_stat(
            "totalRebounds",
            "rebounds",
            "REB"
        )

        fga = get_stat(
            "fieldGoalsAttempted",
            "FGA"
        )

        three_pa = get_stat(
            "threePointFieldGoalsAttempted",
            "3PA"
        )

        result[position_group].append({
            "player": player_name,
            "position": position,
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
# HISTÓRICO DE PARTIDAS DA EQUIPE
# ============================================================

def get_completed_team_games(
    schedule_data,
    team_abbr,
    limit=10
):

    games = []

    for event in schedule_data.get("events", []):

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
# HISTÓRICO DEFENSIVO L5 / L10
# ============================================================

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

    schedule = request_json(
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"basketball/nba/teams/{team_abbr.lower()}/schedule",
        {"season": season_year(target_date)}
    )

    games = get_completed_team_games(
        schedule,
        team_abbr,
        1000
    )

    # Somente partidas anteriores à data selecionada.
    games = [
        game
        for game in games
        if game.get("date", "")[:10]
        < target_date.isoformat()
    ]

    # Máximo de 10 partidas defensivas.
    games = games[:limit]

    results = []

    def fetch_game_defense(game):

        summary = request_json(
            "https://site.api.espn.com/apis/site/v2/sports/"
            "basketball/nba/summary",
            {"event": game["game_id"]}
        )

        stats = extract_team_defense_stats(
            summary,
            team_abbr
        )

        if not stats:
            return None

        return {
            **game,
            **stats
        }

    # Reduzida de 5 para 3 chamadas simultâneas
    # para evitar excesso de requisições à ESPN.
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

    history = {
        "team": team_abbr,
        "games_analyzed": len(results),
        "games": results,
        "averages": calculate_averages(
            results
        )
    }

    # Agora somente L5 e L10.
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
            detail="Não foi possível consultar a agenda da ESPN."
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
                "game_id": event.get(
                    "id"
                ),

                "home": home,
                "away": away,

                "status": (
                    event.get(
                        "status",
                        {}
                    )
                    .get(
                        "type",
                        {}
                    )
                    .get(
                        "description",
                        "Não informado"
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

    if isinstance(node, dict):

        athlete_id = node.get(
            "id"
        )

        name = node.get(
            "fullName"
        )

        if athlete_id and name:

            players[str(athlete_id)] = {
                "id": str(
                    athlete_id
                ),
                "player": name
            }

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
                event.get(
                    "eventId",
                    ""
                )
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
                metadata.get(
                    "gameDate"
                )
                or metadata.get(
                    "date"
                )
            )

            if not game_date:
                continue

            game_date = str(
                game_date
            )[:10]

            if len(game_date) != 10:
                continue

            games[event_id] = {
                "date": game_date,

                "team": (
                    metadata.get(
                        "team"
                    )
                    or {}
                ).get(
                    "abbreviation"
                ),

                "opponent": (
                    metadata.get(
                        "opponent"
                    )
                    or {}
                ).get(
                    "abbreviation"
                ),

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

    return tuple(
        ordered
    )


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
        if game["date"]
        < target_date.isoformat()
    ]

    if len(previous_games) < 3:
        return None

    # A análise individual agora considera no máximo
    # os últimos 10 jogos.
    previous_games = previous_games[:10]

    last5 = previous_games[:5]
    last10 = previous_games[:10]

    minutes_avg = sum(
        game["minutes"]
        for game in last5
    ) / len(last5)

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

    avg5 = average(
        last5
    )

    avg10 = average(
        last10
    )

    # Consistência baseada no L5.
    consistency = sum(
        game[stat_key] >= avg5
        for game in last5
    ) / len(last5)

    # Score adaptado para L5 + L10.
    score = (
        avg5 * 0.60
        + avg10 * 0.40
    ) * (
        0.80
        + consistency * 0.20
    )

    # Médias defensivas do adversário.
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

    return {
        "player": player["player"],
        "team": player["team"],
        "opponent": opponent,

        "last_game_team": last5[0].get(
            "team"
        ),

        "team_changed": bool(
            last5[0].get("team")
            and last5[0]["team"]
            != player["team"]
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

        "games_last5": len(
            last5
        ),

        "games_last10": len(
            last10
        ),

        "opponent_defense_l5_avg": defense_l5,
        "opponent_defense_l10_avg": defense_l10,

        "opponent_defense_games_l5": (
            defense_games_l5
        ),

        "opponent_defense_games_l10": (
            defense_games_l10
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

    # --------------------------------------------------------
    # ELENCOS E AGENDA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # HISTÓRICO DEFENSIVO
    # --------------------------------------------------------

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

    # Reduzido de 4 para 3 equipes simultâneas.
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

    # --------------------------------------------------------
    # HISTÓRICO INDIVIDUAL DOS JOGADORES
    # --------------------------------------------------------

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
                    results.append(
                        result
                    )

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

    # Ordenação pela pontuação.
    results.sort(
        key=lambda item: item["triage_score"],
        reverse=True
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
        "market_label": MARKETS[market]["label"],

        "count": min(
            len(results),
            final_limit
        ),

        "source": "ESPN",
        "season": season_for(
            target_date
        ),

        "teams_found": roster_data["teams_found"],
        "players_found": len(players),

        "players_consulted": successful_logs,
        "players_with_errors": errors,

        "updated_at": datetime.now().isoformat(),

        "method_note": (
            "Triagem estatística baseada exclusivamente na temporada "
            "regular atual da NBA. Médias individuais dos últimos 5 "
            "e 10 jogos anteriores à data selecionada. Filtro mínimo "
            "de 24 minutos médios nos últimos cinco jogos. Inclui "
            "indicadores defensivos do adversário nos períodos L5 "
            "e L10 como contexto complementar. Os indicadores "
            "defensivos não fazem parte da pontuação. Não considera "
            "linhas, odds, lesões ou escalações. A pontuação não "
            "representa probabilidade de acerto."
        ),

        "candidates": results[:final_limit]
    }


# ============================================================
# ENDPOINTS PRINCIPAIS
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "NBA Props Scanner",
        "version": "3.4.0",
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
            detail="Mercado inválido: "
            + ", ".join(
                MARKETS.keys()
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
# ENDPOINTS DE DIAGNÓSTICO
# ============================================================

@app.get("/debug-gamelog")
def debug_gamelog():

    athlete_id = "1966"
    year = 2027

    data = request_json(
        GAMELOG_URL.format(
            athlete=athlete_id
        ),
        {"season": year}
    )

    events = data.get(
        "events"
    )

    if isinstance(events, dict):

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

    elif isinstance(events, list):

        events_sample = events[:3]
        events_count = len(
            events
        )

    else:

        events_sample = events
        events_count = 0

    return {
        "athlete_id": athlete_id,
        "season_year": year,
        "top_level_keys": list(
            data.keys()
        ),
        "categories_present": (
            "categories" in data
        ),
        "labels": data.get(
            "labels"
        ),
        "names": data.get(
            "names"
        ),
        "season_types": data.get(
            "seasonTypes"
        ),
        "events_type": type(
            events
        ).__name__,
        "events_count": events_count,
        "events_sample": events_sample
    }


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
                if game["date"]
                < selected_date
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
        "season_year": year,

        "teams_found": data["teams_found"],
        "players_found": len(players),

        "players_checked": players_checked,
        "players_with_errors": players_with_errors,

        "sample_player": (
            sample_player["player"]
            if sample_player
            else None
        ),

        "sample_player_id": (
            sample_player["id"]
            if sample_player
            else None
        ),

        "sample_games": len(
            previous_games
        ),

        "sample_game": (
            previous_games[0]
            if previous_games
            else None
        ),

        "checked_at": datetime.now().isoformat()
    }


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
            if game["date"]
            < selected_date
        ]

        if not previous_games:

            return {
                "player": player["player"],
                "player_id": player["id"],
                "current_team": player["team"],
                "status": "no_previous_games"
            }

        last_game = previous_games[0]

        historical_teams = sorted({
            game.get("team")
            for game in previous_games[:5]
            if game.get("team")
        })

        last_game_team = last_game.get(
            "team"
        )

        if not last_game_team:

            status = "missing_historical_team"

        elif last_game_team != player["team"]:

            status = "team_mismatch"

        else:

            status = "ok"

        return {
            "player": player["player"],
            "player_id": player["id"],
            "current_team": player["team"],

            "last_game_date": last_game["date"],
            "last_game_team": last_game_team,

            "team_changed": bool(
                last_game_team
                and last_game_team
                != player["team"]
            ),

            "teams_in_last5": historical_teams,

            "multiple_teams_last5": (
                len(historical_teams) > 1
            ),

            "status": status
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
                    "player": player["player"],
                    "player_id": player["id"],
                    "current_team": player["team"],
                    "error": str(error)[:250]
                })

    checked_results.sort(
        key=lambda item: item["player"].casefold()
    )

    errors.sort(
        key=lambda item: item["player"].casefold()
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
        "status": (
            "ok"
            if not errors
            else "partial"
        ),

        "source": "ESPN",
        "date": selected_date,
        "season": season_for(
            target_date
        ),

        "teams_found": data["teams_found"],
        "players_found": len(players),

        "players_excluded_multiple_rosters": (
            data.get(
                "players_excluded_multiple_teams",
                0
            )
        ),

        "players_checked": (
            len(checked_results)
            + len(errors)
        ),

        "players_with_history": (
            len(checked_results)
            - len(no_previous_games)
        ),

        "players_without_history": (
            len(no_previous_games)
        ),

        "players_with_team_mismatch": (
            len(mismatches)
        ),

        "players_missing_historical_team": (
            len(missing_historical_team)
        ),

        "players_with_multiple_teams_last5": (
            len(multiple_teams)
        ),

        "players_with_errors": (
            len(errors)
        ),

        "team_mismatches": mismatches,

        "missing_historical_team": (
            missing_historical_team
        ),

        "multiple_teams_last5": (
            multiple_teams
        ),

        "players_without_history": (
            no_previous_games
        ),

        "errors": errors,

        "checked_at": datetime.now().isoformat()
    }


@app.get("/defense-check")
def defense_check():

    summary = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/summary",
        {"event": "0401909918"}
    )

    return {
        "POR_defense": extract_team_defense_stats(
            summary,
            "POR"
        ),

        "DEN_defense": extract_team_defense_stats(
            summary,
            "DEN"
        )
    }


# ============================================================
# NOVO ENDPOINT:
# DEFESA POR POSIÇÃO
# ============================================================

@app.get("/defense-position-check")
def defense_position_check(
    event: str = Query(...),
    defending_team: str = Query(...)
):

    defending_team = defending_team.upper()

    summary = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/summary",
        {"event": event}
    )

    data = extract_opponent_players_by_position(
        summary,
        defending_team
    )

    return {
        "event": event,
        "defending_team": defending_team,
        "opponent": data["opponent"],
        "positions": data["players"]
    }

@app.get("/defense-position-raw-check")
def defense_position_raw_check(
    event: str = Query(...),
    defending_team: str = Query(...)
):
    defending_team = defending_team.upper()

    summary = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/summary",
        {"event": event}
    )

    teams = summary.get("boxscore", {}).get("teams", [])
    players = summary.get("boxscore", {}).get("players", [])

    return {
        "event": event,
        "defending_team": defending_team,
        "teams_count": len(teams),
        "players_count": len(players),
        "players_raw": players
    }


@app.get("/schedule-check")
def schedule_check():

    target_date = date.today()

    data = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/teams/den/schedule",
        {
            "season": season_year(
                target_date
            )
        }
    )

    events = data.get(
        "events",
        []
    )

    return {
        "top_level_keys": list(
            data.keys()
        ),

        "team": data.get(
            "team",
            {}
        ),

        "events_count": len(
            events
        ),

        "sample_events": events[:2]
    }


@app.get("/defense-history-check")
def defense_history_check():

    target_date = date.today()

    schedule = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/teams/den/schedule",
        {
            "season": season_year(
                target_date
            )
        }
    )

    return {
        "team": "DEN",

        "games": get_completed_team_games(
            schedule,
            "DEN",
            5
        )
    }


@app.get("/defense-game-check")
def defense_game_check():

    summary = request_json(
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/summary",
        {"event": "401909918"}
    )

    return {
        "DEN_defense": extract_team_defense_stats(
            summary,
            "DEN"
        ),

        "POR_defense": extract_team_defense_stats(
            summary,
            "POR"
        )
    }


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
