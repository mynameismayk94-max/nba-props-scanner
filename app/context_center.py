"""Central de Contexto NBA: consultas exploratórias isoladas do Scanner e da Auditoria.

Os dados são atualizados somente por chamada explícita a POST /context/defense/refresh.
As consultas de ranking e equipe leem exclusivamente os snapshots desta tabela.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Callable

from fastapi import APIRouter, HTTPException, Query
from psycopg.types.json import Jsonb


POSITIONS = {"G", "F", "C"}
STAT_KEYS = {
    "points": "points",
    "assists": "assists",
    "rebounds": "rebounds",
    "three_pa": "three_pa",
    "two_pa": "two_pa",
    "fgm": "fgm",
}
STAT_LABELS = {
    "points": "Pontos cedidos",
    "assists": "Assistências cedidas",
    "rebounds": "Rebotes cedidos",
    "three_pa": "Tentativas de 3 pontos cedidas",
    "two_pa": "Tentativas de 2 pontos cedidas",
    "fgm": "Cestas convertidas cedidas",
}


def create_context_router(
    get_database_connection: Callable,
    get_team_defense_history: Callable,
    team_abbreviations,
):
    router = APIRouter(tags=["Central de Contexto NBA"])
    teams = tuple(sorted(set(team_abbreviations)))

    def ensure_table():
        with get_database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS context_defense_snapshots (
                        snapshot_date DATE NOT NULL,
                        team TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (snapshot_date, team)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_context_defense_date
                    ON context_defense_snapshots (snapshot_date)
                    """
                )

    @router.on_event("startup")
    def initialize_context_storage():
        try:
            ensure_table()
        except Exception as error:
            # A indisponibilidade do armazenamento contextual não deve
            # impedir a inicialização do Scanner principal.
            import logging
            logging.getLogger("nba_props_scanner").warning(
                "Central de Contexto: tabela não inicializada: %s", error
            )

    def parse_date(value):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Data inválida. Utilize AAAA-MM-DD."
            )

    def build_venue_windows(history, venue):
        games = [
            game for game in history.get("games", [])
            if game.get("home_away") == venue
        ]
        output = {}
        for window_size in (5, 10):
            sample = games[:window_size]
            positions = {}
            for position in ("G", "F", "C"):
                averages = {}
                stat_games = {}
                appearances = {}
                position_games = [
                    game.get("positions", {}).get(position, {})
                    for game in sample
                    if game.get("positions", {}).get(position, {}).get("players_count", 0) > 0
                ]
                for stat in STAT_KEYS:
                    values = [
                        item.get(stat) for item in position_games
                        if item.get(stat) is not None
                    ]
                    averages[stat] = round(sum(values) / len(values), 1) if values else None
                    stat_games[stat] = len(values)
                    appearances[stat] = sum(
                        item.get("stat_players_count", {}).get(stat, 0)
                        for item in position_games
                    )
                player_counts = [item.get("players_count", 0) for item in position_games]
                positions[position] = {
                    "games_analyzed": len(position_games),
                    "players_per_game_avg": round(sum(player_counts) / len(player_counts), 1) if player_counts else 0,
                    "averages": averages,
                    "stat_games_analyzed": stat_games,
                    "stat_player_appearances": appearances
                }
            output[f"L{window_size}"] = {
                "games_analyzed": len(sample),
                "positions": positions
            }
        return output

    def load_snapshots(target_date):
        try:
            with get_database_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT team, payload, captured_at
                        FROM context_defense_snapshots
                        WHERE snapshot_date = %s
                        ORDER BY team
                        """,
                        (target_date,)
                    )
                    rows = cursor.fetchall()
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"Não foi possível consultar o histórico contextual: {error}"
            )
        return {
            team: {
                "payload": payload,
                "captured_at": captured_at.isoformat() if captured_at else None
            }
            for team, payload, captured_at in rows
        }

    @router.post("/context/defense/refresh")
    def refresh_defense_context(
        selected_date: str = Query(alias="date")
    ):
        target_date = parse_date(selected_date)
        ensure_table()

        results = {}
        errors = {}

        # Limita a concorrência: cada histórico de equipe já consulta jogos
        # em paralelo internamente. A atualização é explícita e nunca ocorre
        # durante a abertura dos painéis de leitura.
        def fetch_team(team):
            return team, get_team_defense_history(team, 40, target_date)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(fetch_team, team): team
                for team in teams
            }
            for future in as_completed(futures):
                team = futures[future]
                try:
                    _, history = future.result()
                    if history and history.get("games_analyzed", 0) > 0:
                        history["venue_windows"] = {
                            venue: build_venue_windows(history, venue)
                            for venue in ("home", "away")
                        }
                        results[team] = history
                    else:
                        errors[team] = "Sem jogos válidos na amostra."
                except Exception as error:
                    errors[team] = str(error)[:300]

        # A captura só é publicada quando todas as equipes foram coletadas.
        # Assim, um refresh parcial nunca mistura snapshots de execuções diferentes.
        complete = len(results) == len(teams) and not errors
        status = "ok" if complete else ("partial" if results else "error")
        teams_saved = 0

        if complete:
            try:
                with get_database_connection() as connection:
                    with connection.cursor() as cursor:
                        cursor.executemany(
                            """
                            INSERT INTO context_defense_snapshots
                                (snapshot_date, team, payload, captured_at)
                            VALUES (%s, %s, %s, NOW())
                            ON CONFLICT (snapshot_date, team)
                            DO UPDATE SET
                                payload = EXCLUDED.payload,
                                captured_at = NOW()
                            """,
                            [
                                (target_date, team, Jsonb(payload))
                                for team, payload in results.items()
                            ]
                        )
                teams_saved = len(results)
            except Exception as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"Falha ao persistir a atualização contextual: {error}"
                )

        if complete:
            message = (
                "Atualização completa. Todas as equipes foram salvas na mesma execução."
            )
        elif results:
            message = (
                "Atualização parcial: nenhum dado novo foi gravado para evitar misturar "
                "capturas. O snapshot completo anterior, se existir, foi preservado."
            )
        else:
            message = (
                "Falha total na coleta: nenhum dado novo foi gravado. "
                "O snapshot completo anterior, se existir, foi preservado."
            )

        return {
            "status": status,
            "date": selected_date,
            "teams_expected": len(teams),
            "teams_collected": len(results),
            "teams_saved": teams_saved,
            "teams_with_errors": len(errors),
            "errors": errors,
            "message": message
        }

    @router.get("/context/defense/rankings")
    def defense_rankings(
        selected_date: str = Query(alias="date"),
        position: str = Query(default="G"),
        stat: str = Query(default="points"),
        window: str = Query(default="L5"),
        venue: str = Query(default="all")
    ):
        target_date = parse_date(selected_date)
        position = position.upper()
        window = window.upper()
        venue = venue.lower()

        if venue not in {"all", "home", "away"}:
            raise HTTPException(status_code=400, detail="Local inválido. Use all, home ou away.")

        if position not in POSITIONS:
            raise HTTPException(status_code=400, detail="Posição inválida. Use G, F ou C.")
        if stat not in STAT_KEYS:
            raise HTTPException(status_code=400, detail="Estatística inválida.")
        if window not in {"L5", "L10"}:
            raise HTTPException(status_code=400, detail="Janela inválida. Use L5 ou L10.")

        stored = load_snapshots(target_date)
        if not stored:
            return {
                "status": "not_ready",
                "date": selected_date,
                "message": "Ainda não há dados contextuais para esta data. Execute a atualização da Central de Contexto.",
                "ranking": []
            }

        if venue != "all" and not any(
            (record["payload"] or {}).get("venue_windows", {}).get(venue)
            for record in stored.values()
        ):
            return {
                "status": "not_ready",
                "date": selected_date,
                "venue": venue,
                "message": "Este snapshot ainda não contém a separação Casa/Fora. Atualize a base contextual para gerar as novas amostras.",
                "ranking": []
            }

        ranking = []
        for team, record in stored.items():
            payload = record["payload"] or {}
            if venue == "all":
                period = payload.get(window, {})
            else:
                period = payload.get("venue_windows", {}).get(venue, {}).get(window, {})
            position_data = period.get("positions", {}).get(position, {})
            value = position_data.get("averages", {}).get(stat)
            games = position_data.get("stat_games_analyzed", {}).get(stat, 0)
            if value is None:
                continue
            ranking.append({
                "team": team,
                "allowed_per_game": value,
                "games_analyzed": games,
                "players_per_game_avg": position_data.get("players_per_game_avg"),
                "captured_at": record["captured_at"]
            })

        # Menor produção cedida = defesa mais forte para a estatística.
        ranking.sort(key=lambda item: item["allowed_per_game"])
        for index, item in enumerate(ranking, start=1):
            item["rank"] = index
            item["defense_label"] = "Mais forte" if index <= max(1, round(len(ranking) * 0.25)) else (
                "Mais vulnerável" if index > round(len(ranking) * 0.75) else "Intermediária"
            )

        return {
            "status": "ok",
            "date": selected_date,
            "position": position,
            "stat": stat,
            "stat_label": STAT_LABELS[stat],
            "window": window,
            "venue": venue,
            "teams_available": len(ranking),
            "ranking": ranking
        }

    @router.get("/context/defense/team/{team}")
    def defense_team_detail(
        team: str,
        selected_date: str = Query(alias="date")
    ):
        target_date = parse_date(selected_date)
        team = team.upper()
        if team not in teams:
            raise HTTPException(status_code=404, detail="Equipe NBA não reconhecida.")

        stored = load_snapshots(target_date)
        record = stored.get(team)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Não há contexto salvo para esta equipe/data. Atualize a Central de Contexto."
            )

        return {
            "status": "ok",
            "date": selected_date,
            "team": team,
            "captured_at": record["captured_at"],
            "history": record["payload"]
        }

    return router
