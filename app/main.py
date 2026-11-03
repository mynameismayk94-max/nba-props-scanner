import logging
from datetime import datetime, date
from functools import lru_cache
from time import sleep
from typing import Optional

import pandas as pd
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from nba_api.stats.endpoints import LeagueGameLog


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nba_props_scanner")

app = FastAPI(
    title="NBA Props Scanner",
    description="API para triagem estatística de mercados NBA.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MARKETS = {
    "points": {"column": "PTS", "label": "Pontos"},
    "assists": {"column": "AST", "label": "Assistências"},
    "rebounds": {"column": "REB", "label": "Rebotes"},
    "threes": {"column": "FG3A", "label": "Tentativas de 3 pontos"},
    "attempts": {"column": "FGA", "label": "Tentativas totais de arremesso"}
}


def season_for(selected_date: date) -> str:
    if selected_date.month >= 10:
        start_year = selected_date.year
    else:
        start_year = selected_date.year - 1

    return f"{start_year}-{str(start_year + 1)[-2:]}"


def parse_minutes(value) -> float:
    if pd.isna(value):
        return 0.0

    try:
        if isinstance(value, (int, float)):
            return float(value)

        value = str(value).strip()

        if ":" in value:
            minutes, seconds = value.split(":")
            return float(minutes) + float(seconds) / 60

        return float(value)

    except (ValueError, TypeError):
        return 0.0


def fetch_espn_games(selected_date: str) -> dict:
    date_compact = selected_date.replace("-", "")

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/scoreboard"
    )

    try:
        response = requests.get(
            url,
            params={"dates": date_compact},
            headers={"User-Agent": "Mozilla/5.0 NBA Props Scanner"},
            timeout=(8, 20)
        )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        logger.exception("Falha ao consultar agenda da ESPN")
        raise HTTPException(
            status_code=503,
            detail=f"Não foi possível consultar a agenda da ESPN: {error}"
        )

    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="A fonte da agenda retornou uma resposta inválida."
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


@lru_cache(maxsize=8)
def get_player_logs(season: str) -> pd.DataFrame:
    """
    Consulta os registros da NBA Stats.
    Duas tentativas, com timeout de 12 segundos cada.
    Somente resultados bem-sucedidos ficam em cache.
    """

    last_error = None

    logger.info("Iniciando consulta NBA Stats para temporada %s", season)

    for attempt in range(1, 3):
        try:
            logger.info(
                "Tentativa %s de 2 para temporada %s",
                attempt,
                season
            )

            endpoint = LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
                timeout=12
            )

            frames = endpoint.get_data_frames()

            if not frames or frames[0].empty:
                raise ValueError("A NBA Stats retornou dados vazios.")

            result = frames[0].copy()

            logger.info(
                "NBA Stats respondeu. Temporada: %s. Registros: %s",
                season,
                len(result)
            )

            return result

        except Exception as error:
            last_error = error

            logger.warning(
                "Falha na tentativa %s de 2 da NBA Stats: %s",
                attempt,
                error
            )

            if attempt < 2:
                sleep(2)

    logger.error(
        "Consulta NBA Stats encerrada sem sucesso. Temporada: %s",
        season
    )

    raise HTTPException(
        status_code=503,
        detail=(
            "A NBA Stats não respondeu após duas tentativas. "
            "A fonte pode estar indisponível ou bloqueando a conexão. "
            f"Detalhe: {last_error}"
        )
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NBA Props Scanner",
        "version": "2.1.0"
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
    """
    Diagnóstico simples da conexão estatística.
    Não gera candidatos; verifica se a fonte respondeu.
    """

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

    logger.info("Solicitado diagnóstico estatístico: %s", season)

    df = get_player_logs(season)

    return {
        "status": "ok",
        "source": "NBA Stats",
        "season": season,
        "records": len(df),
        "players": int(df["PLAYER_NAME"].nunique())
        if "PLAYER_NAME" in df.columns else None,
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
            detail="Mercado inválido. Utilize: " + ", ".join(MARKETS.keys())
        )

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

    logger.info(
        "Consulta de candidatos. Data: %s. Mercado: %s",
        selected_date,
        market
    )

    df = get_player_logs(season).copy()

    required_columns = [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "GAME_DATE",
        "MIN",
        "PTS",
        "AST",
        "REB",
        "FG3A",
        "FGA"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Colunas ausentes na fonte estatística: {missing}"
        )

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"],
        errors="coerce"
    )

    df = df.dropna(subset=["GAME_DATE"])

    df = df[
        df["GAME_DATE"].dt.date < target_date
    ].copy()

    if df.empty:
        return {
            "date": selected_date,
            "market": market,
            "market_label": MARKETS[market]["label"],
            "count": 0,
            "source": "NBA Stats",
            "updated_at": datetime.now().isoformat(),
            "method_note": (
                "Não foram encontrados jogos anteriores à data "
                "selecionada nesta temporada."
            ),
            "candidates": []
        }

    df["MINUTES"] = df["MIN"].apply(parse_minutes)

    stat_column = MARKETS[market]["column"]

    df[stat_column] = pd.to_numeric(
        df[stat_column],
        errors="coerce"
    ).fillna(0)

    candidates_list = []

    for player, group in df.groupby("PLAYER_NAME"):
        group = group.sort_values(
            "GAME_DATE",
            ascending=False
        )

        last5 = group.head(5)
        last10 = group.head(10)
        last20 = group.head(20)

        if len(last5) < 3:
            continue

        minutes_last5 = last5["MINUTES"].mean()

        if minutes_last5 < 24:
            continue

        avg5 = last5[stat_column].mean()
        avg10 = last10[stat_column].mean()
        avg20 = last20[stat_column].mean()

        consistency = (
            last5[stat_column] >= avg5
        ).mean()

        score = (
            avg5 * 0.50
            + avg10 * 0.30
            + avg20 * 0.20
        )

        score = score * (0.80 + consistency * 0.20)

        candidates_list.append({
            "player": player,
            "team": group.iloc[0]["TEAM_ABBREVIATION"],
            "last5_avg": round(float(avg5), 2),
            "last10_avg": round(float(avg10), 2),
            "last20_avg": round(float(avg20), 2),
            "minutes_last5_avg": round(float(minutes_last5), 1),
            "triage_score": round(float(score), 2),
            "games_last5": len(last5),
            "games_last10": len(last10),
            "games_last20": len(last20)
        })

    candidates_list.sort(
        key=lambda item: item["triage_score"],
        reverse=True
    )

    candidates_list = candidates_list[:limit]

    logger.info(
        "Consulta concluída. Mercado: %s. Candidatos: %s",
        market,
        len(candidates_list)
    )

    return {
        "date": selected_date,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "count": len(candidates_list),
        "source": "NBA Stats",
        "updated_at": datetime.now().isoformat(),
        "method_note": (
            "Triagem baseada nas médias dos últimos 5, 10 e 20 jogos, "
            "com filtro de média mínima de 24 minutos nos últimos 5 jogos. "
            "A pontuação não representa probabilidade de acerto nem valor "
            "esperado. Não considera linha da casa, odds, lesões, escalações "
            "ou confronto defensivo."
        ),
        "candidates": candidates_list
    }
