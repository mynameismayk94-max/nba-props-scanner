import unittest
from datetime import date, timedelta

from app.audit_metrics import build_model_comparison


def make_row(day, score=5.0, l5=4.0, l10=3.0, actual=4.0):
    return (
        date(2026, 11, 1) + timedelta(days=day),
        "player-1",
        "Player",
        "AAA",
        "BBB",
        "G",
        score,
        l5,
        l10,
        30.0,
        actual,
        "completed",
    )


class ModelComparisonTests(unittest.TestCase):
    def test_compares_models_on_the_same_sample_without_mutating_score(self):
        rows = [make_row(0), make_row(1, score=6.0)]
        result = build_model_comparison(rows)

        self.assertEqual(result["sample"], 2)
        self.assertEqual(result["models"]["l5"]["sample"], 2)
        self.assertEqual(result["models"]["l10"]["sample"], 2)
        self.assertEqual(result["models"]["blend_60_40"]["mae"], 0.4)
        self.assertEqual(result["models"]["score_current"]["mae"], 1.5)
        self.assertEqual(result["temporal_holdout"]["status"], "collecting")

    def test_temporal_holdout_uses_only_latest_dates(self):
        rows = [make_row(day, actual=4.0) for day in range(14)]
        result = build_model_comparison(rows)
        holdout = result["temporal_holdout"]

        self.assertEqual(holdout["status"], "ready")
        self.assertEqual(holdout["train_dates"], 9)
        self.assertEqual(holdout["test_dates"], 5)
        self.assertEqual(holdout["test_start_date"], "2026-11-10")
        self.assertEqual(holdout["test_end_date"], "2026-11-14")
        self.assertEqual(holdout["models"]["l5"]["sample"], 5)

    def test_incomplete_records_are_excluded_consistently(self):
        rows = [make_row(0), make_row(1, l5=None), make_row(2, score=None)]
        result = build_model_comparison(rows)
        self.assertEqual(result["sample"], 1)
        self.assertTrue(all(
            metric["sample"] == 1
            for metric in result["models"].values()
        ))


if __name__ == "__main__":
    unittest.main()
