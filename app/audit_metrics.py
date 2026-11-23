"""Comparação descritiva e temporal de referências pré-jogo da auditoria.

Não altera o triage_score nem reclassifica snapshots históricos.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable


MINIMUM_TEMPORAL_DATES = 14


def _date_key(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _metrics(rows: list[tuple], model: str) -> dict:
    errors = []
    biases = []
    predictions = []
    for row in rows:
        actual = float(row[10])
        l5 = float(row[7])
        l10 = float(row[8])
        score = float(row[6])
        prediction = {
            "l5": l5,
            "l10": l10,
            "blend_60_40": l5 * 0.60 + l10 * 0.40,
            "score_current": score,
        }[model]
        predictions.append(prediction)
        errors.append(abs(actual - prediction))
        biases.append(prediction - actual)

    count = len(rows)
    if not count:
        return {
            "sample": 0,
            "prediction_average": None,
            "mae": None,
            "bias": None,
        }
    return {
        "sample": count,
        "prediction_average": round(sum(predictions) / count, 4),
        "mae": round(sum(errors) / count, 4),
        "bias": round(sum(biases) / count, 4),
    }



def spearman_correlation(pairs: Iterable[tuple[float, float]]) -> float | None:
    """Spearman rank correlation with average ranks for tied values."""
    values = list(pairs)
    if len(values) < 2:
        return None

    def average_ranks(column_index: int) -> list[float]:
        ordered = sorted(
            enumerate(float(pair[column_index]) for pair in values),
            key=lambda item: item[1]
        )
        ranks = [0.0] * len(values)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            average_rank = ((index + 1) + end) / 2
            for offset in range(index, end):
                ranks[ordered[offset][0]] = average_rank
            index = end
        return ranks

    rank_x = average_ranks(0)
    rank_y = average_ranks(1)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(rank_x, rank_y)
    )
    denominator_x = sum((x - mean_x) ** 2 for x in rank_x)
    denominator_y = sum((y - mean_y) ** 2 for y in rank_y)
    denominator = (denominator_x * denominator_y) ** 0.5
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _window_game_count(row: tuple, index: int) -> int | None:
    try:
        value = row[index]
        return int(value) if value is not None else None
    except (IndexError, TypeError, ValueError):
        return None


def build_model_comparison(rows: Iterable[tuple]) -> dict:
    """Compare available and full L5/L10 windows on identical records.

    Expected row positions follow /validation-history SQL:
    0=date, 6=triage_score, 7=L5, 8=L10, 10=actual_value,
    12=games_last5, 13=games_last10.
    """
    eligible = [
        row for row in rows
        if row[0] is not None
        and row[6] is not None
        and row[7] is not None
        and row[8] is not None
        and row[10] is not None
    ]
    models = ("l5", "l10", "blend_60_40", "score_current")
    overall = {model: _metrics(eligible, model) for model in models}

    full_window_rows = [
        row for row in eligible
        if (_window_game_count(row, 12) or 0) >= 5
        and (_window_game_count(row, 13) or 0) >= 10
    ]
    full_window_models = {
        model: _metrics(full_window_rows, model)
        for model in models
    }

    l5_counts = [_window_game_count(row, 12) for row in eligible]
    l10_counts = [_window_game_count(row, 13) for row in eligible]
    effective_samples = {
        "l5": {
            "records": len(eligible),
            "complete_window": sum(value is not None and value >= 5 for value in l5_counts),
            "partial_window": sum(value is not None and value < 5 for value in l5_counts),
            "unknown_window": sum(value is None for value in l5_counts),
        },
        "l10": {
            "records": len(eligible),
            "complete_window": sum(value is not None and value >= 10 for value in l10_counts),
            "partial_window": sum(value is not None and value < 10 for value in l10_counts),
            "unknown_window": sum(value is None for value in l10_counts),
        },
    }

    dates = sorted({_date_key(row[0]) for row in full_window_rows})
    result = {
        "sample": len(eligible),
        "distinct_dates": len({_date_key(row[0]) for row in eligible}),
        "models": overall,
        "effective_sample_counts": effective_samples,
        "full_window_sample": len(full_window_rows),
        "full_window_distinct_dates": len(dates),
        "full_window_models": full_window_models,
        "temporal_holdout": {
            "status": "collecting",
            "minimum_dates": MINIMUM_TEMPORAL_DATES,
            "dates_available": len(dates),
            "train_dates": 0,
            "test_dates": 0,
            "test_start_date": None,
            "test_end_date": None,
            "models": {},
        },
        "method_note": (
            "A comparação geral inclui todos os registros com valores L5/L10 disponíveis, "
            "inclusive janelas parciais. A comparação de janela completa exige 5/5 jogos "
            "para L5 e 10/10 para L10. O holdout temporal usa somente janelas completas. "
            "O Score é um índice de triagem; seu MAE é apenas diagnóstico numérico."
        ),
    }

    if len(dates) < MINIMUM_TEMPORAL_DATES:
        return result

    split_index = max(1, int(len(dates) * 0.70))
    split_index = min(split_index, len(dates) - 1)
    train_dates = dates[:split_index]
    test_dates = set(dates[split_index:])
    holdout_rows = [
        row for row in full_window_rows if _date_key(row[0]) in test_dates
    ]
    result["temporal_holdout"] = {
        "status": "ready",
        "minimum_dates": MINIMUM_TEMPORAL_DATES,
        "dates_available": len(dates),
        "train_dates": len(train_dates),
        "test_dates": len(test_dates),
        "test_start_date": min(test_dates),
        "test_end_date": max(test_dates),
        "models": {
            model: _metrics(holdout_rows, model)
            for model in models
        },
    }
    return result


def is_temporally_valid_peer(candidate_row: tuple, target_row: tuple) -> bool:
    """Only compare a target with a different player from a strictly earlier date."""
    candidate_date = candidate_row[0]
    target_date = target_row[0]
    if candidate_date is None or target_date is None:
        return False
    return (
        _date_key(candidate_date) < _date_key(target_date)
        and str(candidate_row[1]) != str(target_row[1])
    )



def build_top10_vs_rest_metrics(rows: Iterable[tuple], min_records: int = 30, min_players: int = 15) -> dict:
    """Compare daily top-10 audit ranks with the remaining eligible candidates."""
    completed = [
        row for row in rows
        if row[10] is not None and row[14] is not None
    ]
    top10_rows = [row for row in completed if int(row[14]) <= 10]
    rest_rows = [row for row in completed if int(row[14]) > 10]

    def summarize(group):
        residual_rows = [
            row for row in group
            if row[8] is not None and (_window_game_count(row, 13) or 0) >= 10
        ]
        residuals = [
            float(row[10]) - float(row[8])
            for row in residual_rows
        ]
        return {
            "records": len(group),
            "players": len({str(row[1]) for row in group}),
            "actual_average": (
                round(sum(float(row[10]) for row in group) / len(group), 2)
                if group else None
            ),
            "l10_residual_sample": len(residuals),
            "l10_residual_average": (
                round(sum(residuals) / len(residuals), 2)
                if residuals else None
            ),
        }

    top10 = summarize(top10_rows)
    rest = summarize(rest_rows)
    residual_difference = (
        round(top10["l10_residual_average"] - rest["l10_residual_average"], 2)
        if top10["l10_residual_average"] is not None
        and rest["l10_residual_average"] is not None
        else None
    )
    actual_difference = (
        round(top10["actual_average"] - rest["actual_average"], 2)
        if top10["actual_average"] is not None
        and rest["actual_average"] is not None
        else None
    )
    top10_dates = {
        _date_key(row[0]) for row in top10_rows
        if row[0] is not None and row[8] is not None
        and (_window_game_count(row, 13) or 0) >= 10
    }
    rest_dates = {
        _date_key(row[0]) for row in rest_rows
        if row[0] is not None and row[8] is not None
        and (_window_game_count(row, 13) or 0) >= 10
    }
    dates_count = len(top10_dates & rest_dates)
    volume_sufficient = (
        top10["l10_residual_sample"] >= min_records
        and rest["l10_residual_sample"] >= min_records
        and top10["players"] >= min_players
        and rest["players"] >= min_players
    )
    sample_status = (
        "insufficient" if not volume_sufficient
        else "sufficient" if dates_count >= 14
        else "collecting"
    )
    return {
        "top10": top10,
        "rest": rest,
        "actual_average_difference": actual_difference,
        "l10_residual_difference": residual_difference,
        "distinct_dates": dates_count,
        "minimum_dates_for_validation": 14,
        "minimum_records_each_group": min_records,
        "minimum_players_each_group": min_players,
        "sample_status": sample_status,
        "method_note": (
            "Compara os 10 primeiros candidatos de cada data, ordenados pelo ranking "
            "imutável da captura, com os demais elegíveis. Além da média real, mostra "
            "resultado menos média pré-jogo L10 para reduzir diferenças de nível entre "
            "jogadores. Análise descritiva; não é taxa de acerto nem rentabilidade."
        ),
    }


def build_context_reference_metrics(records: Iterable[dict]) -> dict:
    """Compare descriptive defensive references with the player's own L10 baseline."""
    fields = (
        "opponent_position_minutes_filtered_defense_l10_avg",
        "opponent_position_minute_band_defense_l10_avg",
        "opponent_position_equal_game_defense_l10_avg",
    )
    completed = [
        item for item in records
        if item.get("actual") is not None
        and item.get("player_l10_avg") is not None
    ]

    def number(value):
        try:
            value = float(value)
            return value if value == value and abs(value) != float("inf") else None
        except (TypeError, ValueError):
            return None

    def summarize(field):
        usable = [
            item for item in completed
            if number(item.get(field)) is not None
        ]
        baseline_errors = [
            abs(number(item["actual"]) - number(item["player_l10_avg"]))
            for item in usable
        ]
        context_errors = [
            abs(number(item["actual"]) - number(item[field]))
            for item in usable
        ]
        baseline_bias = [
            number(item["actual"]) - number(item["player_l10_avg"])
            for item in usable
        ]
        context_bias = [
            number(item["actual"]) - number(item[field])
            for item in usable
        ]
        context_values = [number(item[field]) for item in usable]
        return {
            "sample": len(usable),
            "coverage_pct": round(len(usable) / len(completed) * 100, 2) if completed else 0.0,
            "baseline_l10_mae": round(sum(baseline_errors) / len(baseline_errors), 4) if baseline_errors else None,
            "context_mae": round(sum(context_errors) / len(context_errors), 4) if context_errors else None,
            "context_mae_delta_vs_l10": (
                round(sum(context_errors) / len(context_errors) - sum(baseline_errors) / len(baseline_errors), 4)
                if context_errors and baseline_errors else None
            ),
            "baseline_bias": round(sum(baseline_bias) / len(baseline_bias), 4) if baseline_bias else None,
            "context_bias": round(sum(context_bias) / len(context_bias), 4) if context_bias else None,
            "context_average": round(sum(context_values) / len(context_values), 4) if context_values else None,
        }

    return {
        "completed_records": len(completed),
        "references": {field: summarize(field) for field in fields},
        "method_note": (
            "Comparação exploratória entre referências defensivas capturadas antes "
            "do jogo e a média L10 do próprio jogador. Não mede causalidade, "
            "probabilidade de over/under ou valor esperado. A utilidade incremental "
            "deve ser confirmada em holdout temporal quando houver pelo menos 14 datas."
        ),
    }


def build_gap_group_metrics(records: Iterable[dict], period_key: str) -> dict:
    """Summarize defensive groups and player-baseline-adjusted outcomes."""
    baseline_key = "player_l5_avg" if period_key == "l5" else "player_l10_avg"
    group = list(records)
    completed = [item for item in group if item.get("actual") is not None]
    baseline_rows = [
        item for item in completed
        if item.get(baseline_key) is not None
    ]

    def average(values):
        values = [float(value) for value in values if value is not None]
        return round(sum(values) / len(values), 2) if values else None

    def period_values(key, field):
        return [
            (item.get(key) or {}).get(field)
            for item in group
            if (item.get(key) or {}).get(field) is not None
        ]

    return {
        "count": len(group),
        "completed": len(completed),
        "defense_allowed_average": average(period_values(period_key, "defense_allowed")),
        "league_average": average(period_values(period_key, "league_average")),
        "actual_player_average": average([item["actual"] for item in completed]),
        "deviation_pct_average": average(period_values(period_key, "difference_pct")),
        "defense_games_average": average([
            value for value in period_values(period_key, "games") if float(value) > 0
        ]),
        "league_games_average": average([
            value for value in period_values(period_key, "league_games") if float(value) > 0
        ]),
        "baseline_sample": len(baseline_rows),
        "player_baseline_average": average([item[baseline_key] for item in baseline_rows]),
        "actual_minus_player_baseline_average": average([
            float(item["actual"]) - float(item[baseline_key])
            for item in baseline_rows
        ]),
    }
