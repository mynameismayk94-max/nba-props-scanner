"""Motor isolado de contexto de companheiros; não altera Scanner nem triage_score."""
from __future__ import annotations


def _team_abbr(block: dict) -> str:
    return str((block.get("team") or {}).get("abbreviation") or "").strip().upper()


def _athlete_id(entry: dict) -> str | None:
    athlete = entry.get("athlete") or {}
    value = athlete.get("id")
    return str(value).strip() if value is not None else None


def _minutes_value(entry: dict, keys: list[str]) -> tuple[bool, float | None]:
    raw = entry.get("stats", entry.get("statistics"))
    if raw is None:
        return False, None

    if isinstance(raw, dict):
        value = raw.get("minutes", raw.get("MIN"))
        present = "minutes" in raw or "MIN" in raw
    elif isinstance(raw, list):
        index = next((i for i, key in enumerate(keys) if str(key).strip().lower() in {"minutes", "min"}), None)
        present = index is not None and index < len(raw)
        value = raw[index] if present else None
    else:
        return False, None

    if isinstance(value, dict):
        value = value.get("displayValue", value.get("value"))

    if value is None:
        return present, None

    text = str(value).strip()
    if text in {"", "--", "-", "N/A"}:
        return present, None

    try:
        if ":" in text:
            mins, secs = text.split(":", 1)
            return present, float(mins) + float(secs) / 60
        return present, float(text)
    except (TypeError, ValueError):
        return present, None


def _explicit_dnp(entry: dict) -> bool:
    """Só aceita marcadores explícitos de não participação."""
    athlete = entry.get("athlete") or {}
    values = [
        entry.get("didNotPlay"),
        athlete.get("didNotPlay"),
        entry.get("did_not_play"),
        entry.get("status"),
        entry.get("reason"),
        entry.get("displayStatus"),
        entry.get("shortComment")
    ]
    for value in values:
        if value is True:
            return True
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "dnp", "did not play", "did not dress",
                "inactive", "not active", "coach's decision",
                "injury", "illness", "suspended"
            }:
                return True
    return False


def extract_team_player_status(summary: dict, team: str) -> dict[str, dict]:
    """Mapeia atleta -> played/confirmed_absent/unknown a partir do boxscore bruto.

    Ausência de atleta no boxscore nunca é interpretada como DNP.
    """
    team = str(team or "").strip().upper()
    result: dict[str, dict] = {}
    blocks = (summary.get("boxscore") or {}).get("players") or []

    for block in blocks:
        if _team_abbr(block) != team:
            continue
        stat_blocks = block.get("statistics") or []
        if isinstance(stat_blocks, dict):
            stat_blocks = [stat_blocks]

        for stat_block in stat_blocks:
            if not isinstance(stat_block, dict):
                continue
            keys = stat_block.get("keys") or stat_block.get("names") or stat_block.get("labels") or []
            for entry in stat_block.get("athletes") or []:
                if not isinstance(entry, dict):
                    continue
                player_id = _athlete_id(entry)
                if not player_id:
                    continue
                minutes_present, minutes = _minutes_value(entry, keys)
                if minutes is not None and minutes > 0:
                    status = "played"
                    evidence = "positive_minutes"
                elif minutes_present and minutes == 0:
                    status = "confirmed_absent"
                    evidence = "explicit_zero_minutes"
                elif _explicit_dnp(entry):
                    status = "confirmed_absent"
                    evidence = "explicit_dnp_status"
                else:
                    status = "unknown"
                    evidence = "insufficient_boxscore_data"

                # ESPN pode repetir atletas em blocos (estatísticas tradicionais,
                # avançadas etc.). Um bloco posterior sem MIN não pode apagar
                # uma evidência positiva já encontrada.
                priority = {
                    "unknown": 0,
                    "confirmed_absent": 1,
                    "played": 2
                }
                existing = result.get(player_id)
                if (
                    existing is None
                    or priority[status] > priority.get(existing.get("status"), -1)
                ):
                    result[player_id] = {
                        "status": status,
                        "minutes": minutes,
                        "evidence": evidence
                    }

    return result



def build_teammate_shortlist(
    candidate_id: str,
    team: str,
    roster: list[dict],
    player_logs_by_id: dict[str, tuple],
    team_games: list[dict],
    limit: int = 3,
    min_appearances: int = 5,
    min_average_minutes: float = 20.0
) -> list[dict]:
    """Seleciona companheiros relevantes usando apenas dados anteriores.

    team_games deve conter os últimos jogos concluídos da equipe, em ordem
    decrescente. Logs de passagens por outras equipes são descartados.
    """
    team = str(team or "").strip().upper()
    game_ids = {
        str(game.get("game_id"))
        for game in team_games
        if game.get("game_id")
    }
    if not team or not game_ids:
        return []

    candidates = []
    for teammate in roster:
        teammate_id = str(teammate.get("id") or "")
        if not teammate_id or teammate_id == str(candidate_id):
            continue
        player_entry = player_logs_by_id.get(teammate_id)
        if not player_entry:
            continue

        player, raw_logs = player_entry
        relevant_logs = [
            log for log in raw_logs
            if str(log.get("event_id") or "") in game_ids
            and str(log.get("team") or "").upper() == team
            and float(log.get("minutes") or 0) > 0
        ]
        # Deduplica por partida para evitar duplicidade de categorias/linhas.
        by_game = {str(log["event_id"]): log for log in relevant_logs}
        appearances = list(by_game.values())
        if len(appearances) < min_appearances:
            continue

        average_minutes = sum(float(g["minutes"]) for g in appearances) / len(appearances)
        if average_minutes < min_average_minutes:
            continue

        average_points = sum(float(g.get("points") or 0) for g in appearances) / len(appearances)
        average_assists = sum(float(g.get("assists") or 0) for g in appearances) / len(appearances)
        average_fga = sum(float(g.get("fga") or 0) for g in appearances) / len(appearances)
        appearance_rate = len(appearances) / len(game_ids)
        offensive_involvement = average_points + (1.5 * average_assists) + (0.5 * average_fga)
        importance_score = (
            (average_minutes * 0.5)
            + (appearance_rate * 10)
            + (offensive_involvement * 0.25)
        )

        candidates.append({
            "player_id": teammate_id,
            "player": player.get("player") or player.get("name") or teammate.get("player") or "Jogador",
            "position": player.get("position") or teammate.get("position"),
            "team": team,
            "games_in_window": len(appearances),
            "window_games": len(game_ids),
            "appearance_rate": round(appearance_rate, 3),
            "average_minutes": round(average_minutes, 1),
            "average_points": round(average_points, 1),
            "average_assists": round(average_assists, 1),
            "average_fga": round(average_fga, 1),
            "importance_score": round(importance_score, 3),
            "selection_source": "automatic_shortlist"
        })

    candidates.sort(
        key=lambda item: (
            item["importance_score"],
            item["average_minutes"],
            item["games_in_window"]
        ),
        reverse=True
    )
    return candidates[:max(0, min(int(limit), 3))]



def build_candidate_teammate_context(
    candidate_id: str,
    team: str,
    roster: list[dict],
    player_logs_by_id: dict[str, tuple],
    team_games: list[dict],
    teammate_status_by_game: dict[str, dict],
    stat_keys: tuple[str, ...] = (
        "points", "assists", "rebounds", "three_pa",
        "two_pa", "fgm", "minutes"
    )
) -> dict:
    """Monta shortlist e splits para todas as métricas em uma passagem."""
    shortlist = build_teammate_shortlist(
        candidate_id=candidate_id,
        team=team,
        roster=roster,
        player_logs_by_id=player_logs_by_id,
        team_games=team_games,
        limit=3
    )
    comparisons = {}
    for teammate in shortlist:
        teammate_id = teammate["player_id"]
        comparisons[teammate_id] = {
            stat_key: compare_candidate_with_teammate(
                candidate_logs=list(player_logs_by_id.get(str(candidate_id), ({}, ()))[1]),
                team_games=team_games,
                teammate_status_by_game=teammate_status_by_game,
                teammate_id=teammate_id,
                stat_key=stat_key
            )
            for stat_key in stat_keys
        }

    return {
        "algorithm_version": 1,
        "window": "team_last_10_games",
        "team": str(team or "").upper(),
        "shortlist": shortlist,
        "comparisons": comparisons,
        "absence_rule": "explicit_zero_minutes_or_explicit_dnp_only"
    }


def summarize_production(games: list[dict], stat_key: str) -> dict:
    valid = [
        game for game in games
        if game.get("minutes") is not None
        and float(game.get("minutes") or 0) > 0
        and game.get(stat_key) is not None
    ]
    if not valid:
        return {"games": 0, "average": None, "minutes_average": None}
    return {
        "games": len(valid),
        "average": round(sum(float(g[stat_key]) for g in valid) / len(valid), 2),
        "minutes_average": round(sum(float(g["minutes"]) for g in valid) / len(valid), 1)
    }


def compare_candidate_with_teammate(
    candidate_logs: list[dict],
    team_games: list[dict],
    teammate_status_by_game: dict[str, dict],
    teammate_id: str,
    stat_key: str
) -> dict:
    """Compara produção do candidato em jogos com/sem o companheiro.

    Apenas jogos anteriores à data-alvo devem ser fornecidos em team_games.
    Jogos sem status comprovado ficam fora dos dois grupos.
    """
    candidate_by_game = {
        str(game.get("event_id")): game
        for game in candidate_logs
        if game.get("event_id")
        and float(game.get("minutes") or 0) > 0
        and game.get(stat_key) is not None
    }
    played, absent, unknown = [], [], []

    for team_game in team_games:
        game_id = str(team_game.get("game_id") or "")
        candidate_game = candidate_by_game.get(game_id)
        if not candidate_game:
            continue
        status = (teammate_status_by_game.get(game_id) or {}).get(str(teammate_id), {})
        label = status.get("status", "unknown")
        if label == "played":
            played.append(candidate_game)
        elif label == "confirmed_absent":
            absent.append(candidate_game)
        else:
            unknown.append(candidate_game)

    with_teammate = summarize_production(played, stat_key)
    without_teammate = summarize_production(absent, stat_key)
    difference = None
    if with_teammate["average"] is not None and without_teammate["average"] is not None:
        difference = round(without_teammate["average"] - with_teammate["average"], 2)

    return {
        "with_teammate": with_teammate,
        "confirmed_absent": without_teammate,
        "difference_absent_minus_played": difference,
        "unknown_games_excluded": len(unknown),
        "interpretation": "observational_difference",
        "status": "ok" if played or absent else "insufficient_data"
    }
