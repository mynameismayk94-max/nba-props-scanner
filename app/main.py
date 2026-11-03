from datetime import datetime, date
from functools import lru_cache
from time import sleep
from typing import Optional

import pandas as pd
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from nba_api.stats.endpoints import LeagueGameLog


app = FastAPI(
    title="NBA Props Scanner",
    description="API para triagem estatística de mercados NBA.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MARKETS = {
    "points": {
        "column": "PTS",
        "label": "Pontos"
    },
    "assists": {
        "column": "AST",
        "label": "Assistências"
    },
    "rebounds": {
        "column": "REB",
        "label": "Rebotes"
    },
    "threes": {
        "column": "FG3A",
        "label": "Tentativas de 3 pontos"
    },
    "attempts": {
        "column": "FGA",
        "label": "Tentativas totais de arremesso"
    }
}


def season_for(selected_date: date) -> str:
    """
    Identifica a temporada NBA correspondente à data.
    """
    if selected_date.month >= 10:
        start_year = selected_date.year
    else:
        start_year = selected_date.year - 1

    return f"{start_year}-{str(start_year + 1)[-2:]}"


def parse_minutes(value) -> float:
    """
    Converte minutos no formato MM:SS ou numérico.
    """
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
    """
    Consulta a agenda da NBA na ESPN.
    """

    date_compact = selected_date.replace("-", "")

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/nba/scoreboard"
    )

    try:
        response = requests.get(
            url,
            params={"dates": date_compact},
            headers={
                "User-Agent": "Mozilla/5.0 NBA Props Scanner"
            },
            timeout=(8, 20)
        )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
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
    Busca estatísticas dos jogadores na NBA Stats.
    Mantém em cache as consultas realizadas com sucesso.
    """

    last_error = None

    for attempt in range(1, 4):
        try:
            endpoint = LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
                timeout=25
            )

            frames = endpoint.get_data_frames()

            if not frames or frames[0].empty:
                raise ValueError("A NBA Stats não retornou estatísticas.")

            return frames[0].copy()

        except Exception as error:
            last_error = error

            if attempt < 3:
                sleep(2 * attempt)

    raise HTTPException(
        status_code=503,
        detail=(
            "Não foi possível obter as estatísticas na NBA Stats "
            "após 3 tentativas. A fonte pode estar lenta ou "
            f"bloqueando a conexão. Detalhe: {last_error}"
        )
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NBA Props Scanner",
        "version": "2.0.0"
    }


@app.get("/games")
def games(
    selected_date: Optional[str] = Query(
        default=None,
        alias="date"
    )
):
    """
    Retorna os jogos da data selecionada usando a ESPN.
    """

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


@app.get("/candidates")
def candidates(
    selected_date: str = Query(alias="date"),
    market: str = Query(default="points"),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Gera uma lista exploratória de candidatos por mercado,
    considerando médias recentes e minutos.
    """

    if market not in MARKETS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mercado inválido. Utilize: "
                + ", ".join(MARKETS.keys())
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
            detail="Data inválida. Utilize o formato AAAA-MM-DD."
        )

    season = season_for(target_date)

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
        column
        for column in required_columns
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

        # Evita priorizar jogadores com pouca utilização recente.
        if minutes_last5 < 24:
            continue

        avg5 = last5[stat_column].mean()
        avg10 = last10[stat_column].mean()
        avg20 = last20[stat_column].mean()

        consistency = (
            last5[stat_column] >= avg5
        ).mean()

        # Pontuação de triagem, não probabilidade de acerto.
        score = (
            avg5 * 0.50
            + avg10 * 0.30
            + avg20 * 0.20
        )

        score = score * (0.80 + consistency * 0.20)

        team = group.iloc[0]["TEAM_ABBREVIATION"]

        candidates_list.append({
            "player": player,
            "team": team,
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

    return {
        "date": selected_date,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "count": len(candidates_list),
        "source": "NBA Stats",
        "updated_at": datetime.now().isoformat(),
        "method_note": (
            "Triagem estatística baseada nas médias dos últimos "
            "5, 10 e 20 jogos, com filtro de média mínima de "
            "24 minutos nos últimos 5 jogos. A pontuação não "
            "representa probabilidade de acerto nem valor esperado. "
            "Não considera linha da casa, odds, lesões, escalações "
            "ou confronto defensivo."
        ),
        "candidates": candidates_list
    }
