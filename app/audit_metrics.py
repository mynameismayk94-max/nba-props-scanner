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


def build_model_comparison(rows: Iterable[tuple]) -> dict:
    """Compare L5, L10, raw 60/40 blend and current Score on identical rows.

    Expected row positions follow /validation-history's SQL result:
    0=date, 6=triage_score, 7=L5, 8=L10, 10=actual_value.
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

    dates = sorted({_date_key(row[0]) for row in eligible})
    result = {
        "sample": len(eligible),
        "distinct_dates": len(dates),
        "models": overall,
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
            "Comparação na mesma amostra de registros concluídos: L5, L10, "
            "média bruta 60/40 e Score atual. MAE e viés são descritivos. "
            "A validação temporal usa somente datas posteriores como holdout; "
            "não altera nem recalcula o Score salvo."
        ),
    }

    if len(dates) < MINIMUM_TEMPORAL_DATES:
        return result

    split_index = max(1, int(len(dates) * 0.70))
    split_index = min(split_index, len(dates) - 1)
    train_dates = dates[:split_index]
    test_dates = set(dates[split_index:])
    holdout_rows = [
        row for row in eligible if _date_key(row[0]) in test_dates
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
