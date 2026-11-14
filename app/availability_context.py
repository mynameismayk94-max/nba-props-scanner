"""Motor de disponibilidade pré-jogo baseado no relatório atual da ESPN.

A regra é conservadora: somente status explícitos de ausência definitiva
retiram um jogador da lista de elegíveis. Questionable/Day-To-Day/Doubtful
continuam como incerteza e não são tratados como ausência confirmada.
"""
from __future__ import annotations

CONFIRMED_OUT_STATUSES = {
    "out", "inactive", "suspended", "injured reserve", "ir",
    "not active", "did not dress",
}

UNCERTAIN_STATUSES = {
    "questionable", "doubtful", "day-to-day", "day to day",
    "probable", "game time decision", "gtd",
}


def _normalize(value) -> str:
    return " ".join(
        str(value or "").strip().lower().replace("_", " ").replace("-", " ").split()
    )


def classify_injury_status(status: str | None) -> str:
    normalized = _normalize(status)
    if normalized in {_normalize(item) for item in CONFIRMED_OUT_STATUSES}:
        return "confirmed_out"
    if normalized in {_normalize(item) for item in UNCERTAIN_STATUSES}:
        return "uncertain"
    if not normalized:
        return "unknown"
    return "reported_active"


def parse_espn_injuries(payload: dict) -> dict[str, dict]:
    """Converte o relatório league-wide da ESPN em atleta -> disponibilidade."""
    result: dict[str, dict] = {}
    for team_block in payload.get("injuries") or []:
        team = team_block.get("team") or {}
        team_abbr = str(team.get("abbreviation") or "").strip().upper()
        for injury in team_block.get("injuries") or []:
            athlete = injury.get("athlete") or {}
            player_id = athlete.get("id")
            if player_id is None:
                continue
            status = injury.get("status")
            result[str(player_id)] = {
                "player_id": str(player_id),
                "player": athlete.get("displayName") or athlete.get("fullName") or "Jogador",
                "team": team_abbr,
                "status_raw": status,
                "availability": classify_injury_status(status),
                "type": (injury.get("type") or {}).get("name"),
                "detail": injury.get("details"),
                "date": injury.get("date"),
            }
    return result


def enrich_players_with_availability(players: list[dict], injury_index: dict[str, dict]) -> list[dict]:
    """Anexa disponibilidade sem alterar score, posição ou ordem."""
    enriched = []
    for player in players:
        item = dict(player)
        info = injury_index.get(str(player.get("id")))
        item["availability"] = info or {
            "player_id": str(player.get("id") or ""),
            "player": player.get("player") or "Jogador",
            "team": str(player.get("team") or "").upper(),
            "status_raw": None,
            "availability": "no_injury_report",
            "type": None,
            "detail": None,
            "date": None,
        }
        enriched.append(item)
    return enriched


def select_eligible_players(
    results: list[dict],
    injury_index: dict[str, dict],
    limit: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Remove apenas ausências confirmadas e preenche a lista até o limite.

    Questionable/Doubtful/Day-To-Day permanecem elegíveis e carregam
    incerteza para a futura interface.
    """
    eligible = []
    removed = []
    for result in results:
        info = injury_index.get(str(result.get("player_id") or ""))
        if info and info.get("availability") == "confirmed_out":
            removed.append({
                "player_id": str(result.get("player_id") or ""),
                "player": result.get("player"),
                "team": result.get("team"),
                "availability": info,
                "reason": "confirmed_out",
            })
            continue
        item = dict(result)
        item["availability"] = info or {
            "availability": "no_injury_report",
            "status_raw": None,
        }
        eligible.append(item)
        if len(eligible) >= max(1, int(limit)):
            break
    return eligible, removed
