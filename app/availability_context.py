"""Motor de disponibilidade pré-jogo baseado no relatório atual da ESPN.

Somente ausências confirmadas retiram um atleta da lista principal.
A interface recebe apenas os indisponíveis que ocupariam uma vaga no Top 10
original; outros atletas OUT não são expostos como uma lista de lesionados.
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
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())

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

def _availability_for_player(player: dict, injury_index: dict[str, dict]) -> dict:
    info = injury_index.get(str(player.get("player_id") or ""))
    if info:
        return info
    return {
        "player_id": str(player.get("player_id") or ""),
        "player": player.get("player") or "Jogador",
        "team": str(player.get("team") or "").upper(),
        "status_raw": None,
        "availability": "no_injury_report",
        "type": None,
        "detail": None,
        "date": None,
    }

def select_eligible_players(results: list[dict], injury_index: dict[str, dict], limit: int = 10):
    """Retorna exibidos, OUT do Top 10 original, elegíveis fora do corte e todos elegíveis."""
    final_limit = max(1, int(limit))
    initial_top_ids = {str(item.get("player_id") or "") for item in results[:final_limit]}
    eligible_all = []
    removed_from_initial_top = []

    for result in results:
        player_id = str(result.get("player_id") or "")
        info = injury_index.get(player_id)
        if info and info.get("availability") == "confirmed_out":
            if player_id in initial_top_ids:
                removed = dict(result)
                removed["availability"] = info
                removed["reason"] = "confirmed_out"
                removed_from_initial_top.append(removed)
            continue

        item = dict(result)
        item["availability"] = _availability_for_player(result, injury_index)
        eligible_all.append(item)

    return (
        eligible_all[:final_limit],
        removed_from_initial_top,
        eligible_all[final_limit:],
        eligible_all,
    )
