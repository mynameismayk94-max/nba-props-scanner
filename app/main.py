
import logging
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
    version="3.1.0"
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


def request_json(url: str, params=None) -> dict:
    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=(8, 25)
        )
        response.raise_for_status()
                result = response.json()

        print("DEBUG URL:", response.url)
        print("DEBUG STATUS:", response.status_code)
        print("DEBUG TIPO JSON:", type(result).__name__)
        print("DEBUG CHAVES JSON:", list(result.keys())[:30] if isinstance(result, dict) else "Não é objeto")
        print("DEBUG RESPOSTA:", str(result)[:1000])

        return result
    except requests.RequestException as error:
        logger.warning("Falha na consulta ESPN: %s", error)
        raise
    except ValueError as error:
        logger.warning("JSON inválido recebido da ESPN: %s", error)
        raise


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


def fetch_espn_games(selected_date: str) -> dict:
    try:
        data = request_json(
            SCOREBOARD_URL,
            {"dates": selected_date.replace("-", "")}
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Não foi possível consultar a agenda da ESPN."
        )

    games = []

    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        home = None
        away = None

        for competitor in competitions[0].get("competitors", []):
            team = competitor.get("team", {})
            item = {
                "abbr": team.get("abbreviation", ""),
                "name": team.get("displayName", "")
            }

            if competitor.get("homeAway") == "home":
                home = item
            elif competitor.get("homeAway") == "away":
                away = item

        if home and away:
            games.append({
                "game_id": event.get("id"),
                "home": home,
                "away": away,
                "status": (
                    event.get("status", {})
                    .get("type", {})
                    .get("description", "Não informado")
                )
            })

    return {
        "date": selected_date,
        "count": len(games),
        "source": "ESPN",
        "games": games
    }


def collect_roster_players(node, players):
    if isinstance(node, dict):
        athlete_id = node.get("id")
        name = node.get("fullName") or node.get("displayName")

        if athlete_id and name:
            players[str(athlete_id)] = {
                "id": str(athlete_id),
                "player": name
            }

        for value in node.values():
            collect_roster_players(value, players)

    elif isinstance(node, list):
        for value in node:
            collect_roster_players(value, players)


@lru_cache(maxsize=40)
def get_team_roster(team: str) -> tuple:
    data = request_json(ROSTER_URL.format(team=team.lower()))

    players = {}
    collect_roster_players(data, players)

    return tuple(players.values())


def get_players_for_date(selected_date: str) -> dict:
    schedule = fetch_espn_games(selected_date)

    teams = set()

    for game in schedule["games"]:
        teams.add(game["home"]["abbr"])
        teams.add(game["away"]["abbr"])

    players = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(get_team_roster, team): team
            for team in teams
        }

        for future in as_completed(futures):
            team = futures[future]

            try:
                roster = future.result()
                for player in roster:
                    players[player["id"]] = {
                        **player,
                        "team": team
                    }
            except Exception as error:
                logger.warning(
                    "Falha no elenco %s: %s",
                    team,
                    error
                )

    return {
        "schedule": schedule,
        "players": list(players.values()),
        "teams_found": len(teams)
    }


@lru_cache(maxsize=800)
def get_athlete_logs(athlete_id: str, year: int) -> tuple:
    data = request_json(
        GAMELOG_URL.format(athlete=athlete_id),
        {"season": year}
    )

    season_label = f"{year - 1}-{str(year)[-2:]} Regular Season"

    print("DEBUG SEASON LABEL:", season_label)
    print("DEBUG CATEGORIES:", [
        category.get("displayName")
        for category in data.get("categories", [])
    ])

    season_category = None

    for category in data.get("categories", []):
        display_name = category.get("displayName", "")

        if display_name.strip().lower() == season_label.lower():
            season_category = category
            break

    if not season_category:
        return tuple()

    event_dates = data.get("events", {})
    games = {}

    for month in season_category.get("categories", []):
        for event in month.get("events", []):
            event_id = str(event.get("eventId", ""))
            stats = event.get("stats", [])

            if not event_id or len(stats) <= 13:
                continue

            metadata = event_dates.get(event_id, {})
            game_date = metadata.get("gameDate") or metadata.get("date")

            if not game_date:
                continue

            game_date = str(game_date)[:10]

            if len(game_date) != 10:
                continue

            games[event_id] = {
                "date": game_date,
                "minutes": parse_minutes(stats[0]),
                "fga": parse_attempts(stats[1]),
                "three_pa": parse_attempts(stats[3]),
                "rebounds": parse_number(stats[7]),
                "assists": parse_number(stats[8]),
                "points": parse_number(stats[13])
            }

    ordered = sorted(
        games.values(),
        key=lambda game: game["date"],
        reverse=True
    )

    return tuple(ordered)


def build_player_result(player, games, target_date, market):
    previous_games = [
        game for game in games
        if game["date"] < target_date.isoformat()
    ]

    if len(previous_games) < 3:
        return None

    last5 = previous_games[:5]

    minutes_avg = sum(
        game["minutes"] for game in last5
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

    last10 = previous_games[:10]
    last20 = previous_games[:20]

    def average(sample):
        if not sample:
            return 0.0

        return sum(
            game[stat_key] for game in sample
        ) / len(sample)

    avg5 = average(last5)
    avg10 = average(last10)
    avg20 = average(last20)

    consistency = sum(
        game[stat_key] >= avg5 for game in last5
    ) / len(last5)

    score = (
        avg5 * 0.50
        + avg10 * 0.30
        + avg20 * 0.20
    ) * (0.80 + consistency * 0.20)

    return {
        "player": player["player"],
        "team": player["team"],
        "last5_avg": round(avg5, 2),
        "last10_avg": round(avg10, 2),
        "last20_avg": round(avg20, 2),
        "minutes_last5_avg": round(minutes_avg, 1),
        "triage_score": round(score, 2),
        "games_last5": len(last5),
        "games_last10": len(last10),
        "games_last20": len(last20)
    }


def collect_candidates(selected_date, market, limit):
    target_date = datetime.strptime(
        selected_date,
        "%Y-%m-%d"
    ).date()

    year = season_year(target_date)

    roster_data = get_players_for_date(selected_date)
    players = roster_data["players"]

    results = []
    successful_logs = 0
    errors = 0

    def fetch_player(player):
        logs = get_athlete_logs(player["id"], year)
        return player, logs

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(fetch_player, player)
            for player in players
        ]

        for future in as_completed(futures):
            try:
                player, logs = future.result()
                successful_logs += 1

                result = build_player_result(
                    player,
                    logs,
                    target_date,
                    market
                )

                if result:
                    results.append(result)

            except Exception as error:
                errors += 1
                logger.warning(
                    "Falha ao consultar jogador: %s",
                    error
                )

    results.sort(
        key=lambda item: item["triage_score"],
        reverse=True
    )

    return {
        "date": selected_date,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "count": min(len(results), limit),
        "source": "ESPN",
        "season": season_for(target_date),
        "teams_found": roster_data["teams_found"],
        "players_found": len(players),
        "players_consulted": successful_logs,
        "players_with_errors": errors,
        "updated_at": datetime.now().isoformat(),
        "method_note": (
            "Triagem estatística baseada exclusivamente na temporada "
            "regular atual da NBA. Médias dos últimos 5, 10 e 20 jogos "
            "anteriores à data selecionada. Filtro mínimo de 24 minutos "
            "médios nos últimos cinco jogos. Não considera linhas, odds, "
            "lesões, escalações ou confronto defensivo. A pontuação não "
            "representa probabilidade de acerto."
        ),
        "candidates": results[:limit]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NBA Props Scanner",
        "version": "3.1.0",
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
        datetime.strptime(selected_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize AAAA-MM-DD."
        )

    return fetch_espn_games(selected_date)



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
            detail="Data inválida. Utilize AAAA-MM-DD."
        )

    data = get_players_for_date(selected_date)
    players = data["players"]

    if not players:
        raise HTTPException(
            status_code=503,
            detail="Nenhum jogador encontrado nos elencos consultados."
        )

    year = season_year(target_date)

    sample_player = None
    previous_games = []
    players_checked = 0
    players_with_errors = 0

    # Testa até 60 jogadores, ignorando quem ainda não tem jogos.
    for player in players[:60]:
        players_checked += 1

        try:
            sample_logs = get_athlete_logs(
                player["id"],
                year
            )

            player_games = [
                game for game in sample_logs
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
        "status": "ok" if sample_player else "warning",
        "source": "ESPN",
        "season": season_for(target_date),
        "season_year": year,
        "teams_found": data["teams_found"],
        "players_found": len(players),
        "players_checked": players_checked,
        "players_with_errors": players_with_errors,
        "sample_player": sample_player["player"] if sample_player else None,
        "sample_player_id": sample_player["id"] if sample_player else None,
        "sample_games": len(previous_games),
        "sample_game": previous_games[0] if previous_games else None,
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
            detail="Mercado inválido: " + ", ".join(MARKETS.keys())
        )

    try:
        datetime.strptime(selected_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Data inválida. Utilize AAAA-MM-DD."
        )

    logger.info(
        "Candidatos: data=%s mercado=%s",
        selected_date,
        market
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
        logger.exception("Erro ao gerar candidatos")
        raise HTTPException(
            status_code=503,
            detail=f"Não foi possível gerar candidatos: {error}"
        )
