"""Motor conservador de disponibilidade pré-jogo baseado no relatório ESPN."""
from __future__ import annotations

import re
import unicodedata

CONFIRMED_OUT_STATUSES = {
    "out", "inactive", "suspended", "injured reserve", "ir",
    "not active", "did not dress",
}
UNCERTAIN_STATUSES = {
    "questionable", "doubtful", "day to day", "probable",
    "game time decision", "gtd",
}

def _normalize(value) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())

def classify_injury_status(status) -> str:
    if isinstance(status, dict):
        status = status.get("name") or status.get("description") or status.get("type")
    normalized = _normalize(status)
    if normalized in {_normalize(item) for item in CONFIRMED_OUT_STATUSES}:
        return "confirmed_out"
    if normalized in {_normalize(item) for item in UNCERTAIN_STATUSES}:
        return "uncertain"
    if not normalized:
        return "unknown"
    return "reported_active"

def _record_injury(result: dict, team_abbr: str, injury: dict, athlete: dict | None = None):
    athlete = athlete or injury.get("athlete") or {}
    player_id = athlete.get("id") or injury.get("athleteId") or injury.get("playerId") or injury.get("id")
    if player_id is None:
        return
    status = injury.get("status")
    if status is None and isinstance(injury.get("availability"), dict):
        status = injury["availability"].get("status")
    record = {
        "player_id": str(player_id),
        "player": athlete.get("displayName") or athlete.get("fullName") or injury.get("displayName") or injury.get("name") or "Jogador",
        "team": str(team_abbr or (athlete.get("team") or {}).get("abbreviation") or "").strip().upper(),
        "status_raw": status,
        "availability": classify_injury_status(status),
        "type": (injury.get("type") or {}).get("name") if isinstance(injury.get("type"), dict) else injury.get("type"),
        "detail": injury.get("details") or injury.get("detail") or injury.get("shortComment") or injury.get("longComment"),
        "date": injury.get("date") or injury.get("updated"),
    }
    # Para entradas repetidas, conserva a classificação mais restritiva.
    previous = result.get(str(player_id))
    priority = {"confirmed_out": 3, "uncertain": 2, "unknown": 1, "reported_active": 0}
    if previous is None or priority[record["availability"]] >= priority[previous["availability"]]:
        result[str(player_id)] = record

def parse_espn_injuries(payload: dict) -> dict[str, dict]:
    """Aceita o formato agrupado da ESPN e variantes achatadas por atleta."""
    if not isinstance(payload, dict):
        raise ValueError("Relatório de lesões ESPN não é um objeto JSON.")

    result: dict[str, dict] = {}
    if isinstance(payload.get("injuries"), list):
        for block in payload["injuries"]:
            if not isinstance(block, dict):
                continue
            # Formato agrupado: injuries[] contém team + injuries[].
            if isinstance(block.get("injuries"), list):
                team = block.get("team") or {}
                team_abbr = team.get("abbreviation") or block.get("teamAbbreviation") or ""
                for injury in block["injuries"]:
                    if isinstance(injury, dict):
                        _record_injury(result, team_abbr, injury)
            # Formato achatado: injuries[] já contém atleta/status.
            elif block.get("athlete") or block.get("athleteId") or block.get("playerId"):
                team = block.get("team") or {}
                _record_injury(result, team.get("abbreviation") or block.get("teamAbbreviation") or "", block)

    # Algumas respostas/serializações expõem atletas com injuries próprios.
    athletes = payload.get("athletes")
    if isinstance(athletes, list):
        for athlete_row in athletes:
            if not isinstance(athlete_row, dict):
                continue
            athlete = athlete_row.get("athlete") or athlete_row
            team = athlete.get("team") or athlete_row.get("team") or {}
            for injury in athlete_row.get("injuries") or athlete.get("injuries") or []:
                if isinstance(injury, dict):
                    _record_injury(result, team.get("abbreviation") or "", injury, athlete)

    # Não confundir um JSON válido, mas com formato desconhecido, com zero lesões.
    known_shape = isinstance(payload.get("injuries"), list) or isinstance(athletes, list)
    if not known_shape:
        raise ValueError("Formato inesperado do relatório ESPN: campos injuries/athletes ausentes.")
    return result

def _name_key(value) -> str:
    return _normalize(value)

def _find_player_injury(player: dict, injury_index: dict[str, dict]):
    player_id = str(player.get("player_id") or "")
    info = injury_index.get(player_id)
    if info:
        return info
    # Fallback estrito por nome completo, usado apenas quando o ID não casou.
    target_name = _name_key(player.get("player"))
    if not target_name:
        return None
    matches = [item for item in injury_index.values() if _name_key(item.get("player")) == target_name]
    if len(matches) == 1:
        return matches[0]
    return None

def _availability_for_player(player: dict, injury_index: dict[str, dict]) -> dict:
    info = _find_player_injury(player, injury_index)
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
        info = _find_player_injury(result, injury_index)
        if info and info.get("availability") == "confirmed_out":
            if str(result.get("player_id") or "") in initial_top_ids:
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
