
import logging
import re
from datetime import datetime, date
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nba_props_scanner")

VERSION = "3.0.0"

app = FastAPI(
    title="NBA Props Scanner",
    description="API de triagem estatística NBA com dados ESPN.",
    version=VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 NBA Props Scanner"
}

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/nba/scoreboard"
)

ESPN_ROSTER = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/nba/teams/{team}/roster"
)

ESPN_GAMELOG = (
    "https://site.web.api.espn.com/apis/common/v3/sports/"
    "basketball/nba/athletes/{athlete_id}/gamelog"
)

MARKETS = {
    "points": {
        "label": "Pontos",
        "index": 13
    },
    "assists": {
        "label": "Assistências",
        "index": 8
    },
    "rebounds": {
        "label": "Rebotes",
        "index": 7
    },
    "threes": {
        "label": "Tentativas de 3 pontos",
        "index": 3,
        "attempts": True
    },
    "attempts": {
        "label": "Tentativas totais de arremesso",
        "index": 1,
        "attempts": True
    }
}


def season_for(selected_date: date) -> str:
    if selected_date.month >= 10:
        start_year = selected_date.year
    else:
        start_year = selected_date.year - 1

    return f"{start_year}-{str(start_year + 1)[-2:]}"


def espn_season_year(season: str) -> int:
    return int(season.split("-")[0]) + 1


def parse_number(value) -> float:
    try:
        if value is None:
            return 0.0

        return float(str(value).strip())

    except (ValueError, TypeError):
        return 0.0


def parse_attempts(value) -> float:
    """
    ESPN apresenta arremessos no formato:
    14-24 = 14 convertidos em 24 tentativas.
    """

    if not value:
        return 0.0

    parts = str(value).split("-")

    if len(parts) != 2:
        return 0.0

    return parse_number(parts[1])


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


def espn_get(url: str, params=None) -> dict:
    try:
        response = requests.get(
            url,
            params=params,
            headers=ESPN_HEADERS,
            timeout=(8, 20)
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        logger.warning("Falha ESPN: %s - %s", url, error)
        raise

    except ValueError as error:
        logger.warning("JSON inválido da ESPN: %s", error)
        raise


def fetch_espn_games(selected_date: str) -> dict:
    try:
        data = espn_get(
            ESPN_SCOREBOARD,
            params={"dates": selected_date.replace("-", "")}
        )

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Não foi possível consultar a agenda da ESPN: {error}"
        )

    games = []

    for event in data.get("events", []):
        competitions = event.get("competitions", [])

        if not competitions:
            continue

        competitors = competitions[0].get("competitors", [])
        home = None
        away = None

        for competitor in competitors:
            team = competitor.get("team", {})

            team_data = {
                "abbr": team.get("abbreviation", ""),
                "name": team.get("displayName", "")
            }

            if competitor.get("homeAway") == "home":
                home = team_data

            elif competitor.get("homeAway") == "away":
                away = team_data

        if not home or not away:
            continue

        status = (
            event.get("status", {})
            .get("type", {})
            .get("description", "Status não informado")
        )

        games.append({
            "game_id": event.get("id"),
            "away": away,
            "home": home,
            "status": status
        })

    return {
        "date": selected_date,
        "count": len(games),
        "source": "ESPN",
        "games": games
    }


@lru_cache(maxsize=40)
def get_team_roster(team: str) -> tuple:
    """
    Consulta o elenco de uma equipe.
    O resultado fica em cache para evitar consultas repetidas.
    """

    try:
        data = espn_get(
            ESPN_ROSTER.format(team=team.lower())
        )

    except Exception:
        logger.exception("Falha ao consultar elenco: %s", team)
        return tuple()

    athletes = {}
    
    def walk(obj):
        if isinstance(obj, dict):
            athlete_id = obj.get("id")
            name = obj.get("fullName") or obj.get("displayName")

            if athlete_id and name:
                athlete_id = str(athlete_id)

                if athlete_id.isdigit():
                    athletes[athlete_id] = {
                        "id": athlete_id,
                        "name": name,
                        "team": team.upper()
                    }

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    result = tuple(athletes.values())

    logger.info(
        "Elenco %s: %s atletas encontrados",
        team,
        len(result)
    )

    return result


@lru_cache(maxsize=800)
def get_athlete_logs(
    athlete_id: str,
    season: str
) -> tuple:
    """
    Consulta o histórico de um atleta na temporada selecionada.
    Somente a Regular Season é considerada.
    """

    season_year = espn_season_year(season)

    try:
        data = espn_get(
            ESPN_GAMELOG.format(athlete_id=athlete_id),
            params={"season": season_year}
        )

    except Exception:
        logger.warning(
            "Não foi possível consultar o atleta %s",
            athlete_id
        )
        return tuple()

    event_dates = data.get("events", {})
    regular_category = None

    for category in data.get("categories", []):
        display_name = str(category.get("displayName", ""))

        if (
            "Regular Season" in display_name
            and season in display_name
        ):
            regular_category = category
            break

    if regular_category is None:
        return tuple()

    games = []

    for month in regular_category.get("categories", []):
        for event in month.get("events", []):
            event_id = str(event.get("eventId", ""))
            stats = event.get("stats", [])

            metadata = event_dates.get(event_id, {})
            game_date = (
                metadata.get("gameDate")
                or metadata.get("date")
            )

            if not game_date or len(stats) < 14:
                continue

            try:
                parsed_date = datetime.strptime(
                    str(game_date)[:10],
                    "%Y-%m-%d"
                ).date()
            except ValueError:
                continue

            games.append({
                "game_id": event_id,
                "game_date": parsed_date.isoformat(),
                "date_object": parsed_date,
                "minutes": parse_minutes(stats[0]),
                "fga": parse_attempts(stats[1]),
                "three_pa": parse_attempts(stats[3]),
                "rebounds": parse_number(stats[7]),
                "assists": parse_number(stats[8]),
                "points": parse_number(stats[13])
            })

    games.sort(
        key=lambda item: item["date_object"],
        reverse=True
    )

    return tuple(games)


def get_players_for_date(selected_date: str) -> tuple:
    """
    Busca os jogadores dos times que disputam a rodada.
    """

    schedule = fetch_espn_games(selected_date)

    teams = set()

    for game in schedule["games"]:
        teams.add(game["home"]["abbr"])
        teams.add(game["away"]["abbr"])

    if not teams:
        return tuple(), schedule

    players = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(get_team_roster, team): team
            for team in sorted(teams)
        }

        for future in as_completed(futures):
            try:
                roster = future.result()

                for athlete in roster:
                    players[athlete["id"]] = athlete

            except Exception:
                logger.exception("Erro ao processar elenco")

    return tuple(players.values()), schedule


def get_market_value(game: dict, market: str) -> float:
    if market == "points":
        return game["points"]

    if market == "assists":
        return game["assists"]

    if market == "rebounds":
        return game["rebounds"]

    if market == "threes":
        return game["three_pa"]

    if market == "attempts":
        return game["fga"]

    return 0.0


def build_candidates(
    selected_date: str,
    market: str,
    limit: int
) -> dict:
    target_date = datetime.strptime(
        selected_date,
        "%Y-%m-%d"
    ).date()

    season = season_for(target_date)

    players, schedule = get_players_for_date(selected_date)

    if not players:
        return {
            "date": selected_date,
            "market": market,
            "market_label": MARKETS[market]["label"],
            "count": 0,
            "source": "ESPN",
            "updated_at": datetime.now().isoformat(),
            "method_note": (
                "Nenhum atleta foi encontrado nos elencos "
                "das equipes da data selecionada."
            ),
            "candidates": []
        }

    candidates_list = []

    logger.info(
        "Consultando %s atletas para %s",
        len(players),
        selected_date
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                get_athlete_logs,
                player["id"],
                season
            ): player
            for player in players
        }

        for future in as_completed(futures):
            player = futures[future]

            try:
                logs = future.result()
            except Exception:
                continue

            eligible = [
                game for game in logs
                if game["date_object"] < target_date
            ]

            eligible.sort(
                key=lambda item: item["date_object"],
                reverse=True
            )

            if not eligible:
                continue

            last5 = eligible[:5]
            last10 = eligible[:10]
            last20 = eligible[:20]

            if len(last5) < 3:
                continue

            minutes_avg = sum(
                game["minutes"] for game in last5
            ) / len(last5)

            if minutes_avg < 24:
                continue

            values5 = [
                get_market_value(game, market)
                for game in last5
            ]

            values10 = [
                get_market_value(game, market)
                for game in last10
            ]

            values20 = [
                get_market_value(game, market)
                for game in last20
            ]

            avg5 = sum(values5) / len(values5)
            avg10 = sum(values10) / len(values10)
            avg20 = sum(values20) / len(values20)

            consistency = sum(
                value >= avg5 for value in values5
            ) / len(values5)

            score = (
                avg5 * 0.50
                + avg10 * 0.30
                + avg20 * 0.20
            )

            score *= 0.80 + consistency * 0.20

            candidates_list.append({
                "player": player["name"],
                "player_id": player["id"],
                "team": player["team"],
                "last5_avg": round(avg5, 2),
                "last10_avg": round(avg10, 2),
                "last20_avg": round(avg20, 2),
                "minutes_last5_avg": round(minutes_avg, 1),
                "triage_score": round(score, 2),
                "games_last5": len(last5),
                "games_last10": len(last10),
                "games_last20": len(last20)
            })

    candidates_list.sort(
        key=lambda item: item["triage_score"],
        reverse=True
    )

    candidates_list = candidates_list[:limit]

    return {
        "date": selected_date,
        "season": season,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "count": len(candidates_list),
        "source": "ESPN",
        "updated_at": datetime.now().isoformat(),
        "teams_in_schedule": len({
            team
            for game in schedule["games"]
            for team in (
                game["home"]["abbr"],
                game["away"]["abbr"]
            )
        }),
        "players_consulted": len(players),
        "method_note": (
            "Triagem baseada exclusivamente na temporada regular "
            f"{season}, com jogos anteriores à data selecionada. "
            "L5, L10 e L20 representam os últimos jogos disponíveis. "
            "Tentativas de arremesso são extraídas do segundo número "
            "do formato acertos-tentativas da ESPN. "
            "Filtro mínimo de 3 jogos em L5 e média de 24 minutos. "
            "A pontuação não representa probabilidade de acerto nem "
            "valor esperado. Não considera linha da casa, odds, "
            "lesões, escalações ou confronto defensivo."
        ),
        "candidates": candidates_list
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NBA Props Scanner",
        "version": VERSION,
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
        parsed_date = datetime.strptime(
            selected_date,
            "%Y-%m-%d"
        ).date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize o formato AAAA-MM-DD."
        )

    return fetch_espn_games(parsed_date.isoformat())


@app.get("/stats-check")
def stats_check(
    selected_date: str = Query(alias="date")
):
    try:
        target_date = datetime.strptime(
            selected_date,
            "%Y-%m-%d"
        ).date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize o formato AAAA-MM-DD."
        )

    season = season_for(target_date)

    players, schedule = get_players_for_date(
        target_date.isoformat()
    )

    if not players:
        raise HTTPException(
            status_code=503,
            detail=(
                "A ESPN não retornou atletas para as equipes "
                "da data selecionada."
            )
        )

    sample = players[0]

    logs = get_athlete_logs(
        sample["id"],
        season
    )

    return {
        "status": "ok" if logs else "warning",
        "source": "ESPN",
        "season": season,
        "season_year": espn_season_year(season),
        "teams_found": len({
            team
            for game in schedule["games"]
            for team in (
                game["home"]["abbr"],
                game["away"]["abbr"]
            )
        }),
        "players_found": len(players),
        "sample_player": sample["name"],
        "sample_player_id": sample["id"],
        "sample_games": len(logs),
        "sample_game": logs[0]["game_date"] if logs else None,
        "checked_at": datetime.now().isoformat()
    }


@app.get("/candidates")
def candidates(
    selected_date: str = Query(alias="date"),
    market: str = Query(default="points"),
    limit: int = Query(default=20, ge=1, le=100)
):
    if market not in MARKETS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mercado inválido. Utilize: "
                + ", ".join(MARKETS.keys())
            )
        )

    try:
        datetime.strptime(selected_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize o formato AAAA-MM-DD."
        )

    logger.info(
        "Consulta ESPN. Data: %s. Mercado: %s",
        selected_date,
        market
    )

    return build_candidates(
        selected_date,
        market,
        limit
    )
