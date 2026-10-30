
from datetime import date, datetime
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2

app = FastAPI(title="NBA Props Scanner", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MARKETS = {
    "points": ("PTS", "Pontos"),
    "assists": ("AST", "Assistências"),
    "rebounds": ("REB", "Rebotes"),
    "threes": ("FG3A", "Tentativas de 3"),
    "attempts": ("FGA", "Tentativas totais"),
}


def season_for(selected_date: date) -> str:
    start_year = selected_date.year if selected_date.month >= 10 else selected_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def parse_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    try:
        if ":" in str(value):
            minutes, seconds = str(value).split(":")
            return float(minutes) + float(seconds) / 60
        return float(value)
    except (ValueError, TypeError):
        return 0.0


@lru_cache(maxsize=8)
def get_player_logs(season: str):
    try:
        endpoint = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
            timeout=30,
        )
        frames = endpoint.get_data_frames()
        if not frames or frames[0].empty:
            return pd.DataFrame()
        return frames[0]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Não foi possível consultar os dados da NBA: {str(exc)[:250]}",
        )


@app.get("/health")
def health():
    return {"status": "ok", "service": "NBA Props Scanner"}


@app.get("/games")
def games(selected_date: date = Query(alias="date")):
    try:
        board = ScoreboardV2(
            game_date=selected_date.strftime("%m/%d/%Y"),
            timeout=30,
        )
        frames = board.get_data_frames()
        headers = frames[0]
        lines = frames[1] if len(frames) > 1 else pd.DataFrame()

        results = []
        if headers.empty:
            return {
                "date": str(selected_date),
                "count": 0,
                "games": [],
                "source": "NBA Stats",
            }

        for _, game in headers.iterrows():
            game_id = str(game.get("GAME_ID", ""))
            home_id = str(game.get("HOME_TEAM_ID", ""))
            away_id = str(game.get("VISITOR_TEAM_ID", ""))

            home = lines[lines["TEAM_ID"].astype(str) == home_id] if not lines.empty else pd.DataFrame()
            away = lines[lines["TEAM_ID"].astype(str) == away_id] if not lines.empty else pd.DataFrame()

            def team_info(frame):
                if frame.empty:
                    return {"abbr": "N/D", "name": "Não disponível"}
                row = frame.iloc[0]
                return {
                    "abbr": str(row.get("TEAM_ABBREVIATION", "N/D")),
                    "name": str(row.get("TEAM_NAME", "Não disponível")),
                }

            results.append({
                "game_id": game_id,
                "home": team_info(home),
                "away": team_info(away),
                "status": str(game.get("GAME_STATUS_TEXT", "Agendado")),
            })

        return {
            "date": str(selected_date),
            "count": len(results),
            "games": results,
            "source": "NBA Stats",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erro ao consultar a agenda: {str(exc)[:250]}",
        )


@app.get("/candidates")
def candidates(
    selected_date: date = Query(alias="date"),
    market: str = Query("points"),
    limit: int = Query(20, ge=1, le=100),
):
    if market not in MARKETS:
        raise HTTPException(
            status_code=400,
            detail=f"Mercado inválido. Escolha: {', '.join(MARKETS)}",
        )

    stat_column, market_label = MARKETS[market]
    season = season_for(selected_date)
    logs = get_player_logs(season)

    if logs.empty:
        return {
            "date": str(selected_date),
            "market": market,
            "market_label": market_label,
            "count": 0,
            "candidates": [],
            "method_note": "A fonte não retornou jogos para esta temporada.",
        }

    required = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
        "GAME_DATE", "MIN", stat_column,
    ]
    missing = [column for column in required if column not in logs.columns]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"A resposta da fonte não contém os campos esperados: {missing}",
        )

    data = logs.copy()
    data["GAME_DATE_PARSED"] = pd.to_datetime(data["GAME_DATE"], errors="coerce")
    data = data[data["GAME_DATE_PARSED"].dt.date <= selected_date]
    data["MINUTES"] = data["MIN"].apply(parse_minutes)
    data[stat_column] = pd.to_numeric(data[stat_column], errors="coerce").fillna(0)
    data = data.sort_values("GAME_DATE_PARSED", ascending=False)

    results = []

    for player_id, group in data.groupby("PLAYER_ID"):
        recent5 = group.head(5)
        recent10 = group.head(10)
        recent20 = group.head(20)

        if len(recent5) < 3:
            continue

        avg5 = float(recent5[stat_column].mean())
        avg10 = float(recent10[stat_column].mean())
        avg20 = float(recent20[stat_column].mean())
        minutes5 = float(recent5["MINUTES"].mean())

        if minutes5 < 24:
            continue

        consistency = float(
            (recent5[stat_column] >= avg5 * 0.75).mean()
        ) if avg5 > 0 else 0

        score = (
            avg5 * 0.50
            + avg10 * 0.30
            + avg20 * 0.20
        ) * min(minutes5 / 30, 1.15) * (0.8 + consistency * 0.2)

        last_row = group.iloc[0]

        results.append({
            "player": str(last_row["PLAYER_NAME"]),
            "team": str(last_row["TEAM_ABBREVIATION"]),
            "last5_avg": round(avg5, 2),
            "last10_avg": round(avg10, 2),
            "last20_avg": round(avg20, 2),
            "minutes_last5_avg": round(minutes5, 1),
            "triage_score": round(score, 2),
            "games_sample": len(recent5),
        })

    results.sort(key=lambda item: item["triage_score"], reverse=True)

    return {
        "date": str(selected_date),
        "market": market,
        "market_label": market_label,
        "count": min(len(results), limit),
        "candidates": results[:limit],
        "method_note": (
            "Triagem estatística baseada nas médias dos últimos 5, 10 e 20 jogos "
            "e minutos recentes. Não considera defesa por posição, lesões, "
            "escalações, linha ou odd da Bet365. Prioridade técnica não é "
            "probabilidade de acerto nem recomendação de aposta."
        ),
        "source": "NBA Stats",
        "season": season,
    }
