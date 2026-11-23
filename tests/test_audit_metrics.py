import unittest
from datetime import date, timedelta

from app.audit_metrics import build_gap_group_metrics, build_model_comparison, is_temporally_valid_peer


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

    def test_gap_metrics_adjust_actual_by_matching_player_baseline(self):
        records = [
            {
                "actual": 12,
                "player_l5_avg": 10,
                "player_l10_avg": 9,
                "l5": {"defense_allowed": 50, "league_average": 45, "difference_pct": 11.1, "games": 5, "league_games": 150},
                "l10": {"defense_allowed": 48, "league_average": 45, "difference_pct": 6.7, "games": 10, "league_games": 300},
            },
            {
                "actual": 7,
                "player_l5_avg": 8,
                "player_l10_avg": 8,
                "l5": {"defense_allowed": 40, "league_average": 45, "difference_pct": -11.1, "games": 5, "league_games": 150},
                "l10": {"defense_allowed": 42, "league_average": 45, "difference_pct": -6.7, "games": 10, "league_games": 300},
            },
        ]
        l5 = build_gap_group_metrics(records, "l5")
        l10 = build_gap_group_metrics(records, "l10")

        self.assertEqual(l5["baseline_sample"], 2)
        self.assertEqual(l5["player_baseline_average"], 9.0)
        self.assertEqual(l5["actual_minus_player_baseline_average"], 0.5)
        self.assertEqual(l10["player_baseline_average"], 8.5)
        self.assertEqual(l10["actual_minus_player_baseline_average"], 1.0)

    def test_similarity_peers_must_be_other_players_from_prior_dates(self):
        target = make_row(5)
        prior_other_player = list(make_row(4))
        prior_other_player[1] = "player-2"
        same_date_other_player = list(make_row(5))
        same_date_other_player[1] = "player-2"
        future_other_player = list(make_row(6))
        future_other_player[1] = "player-2"
        prior_same_player = list(make_row(4))

        self.assertTrue(is_temporally_valid_peer(tuple(prior_other_player), target))
        self.assertFalse(is_temporally_valid_peer(tuple(same_date_other_player), target))
        self.assertFalse(is_temporally_valid_peer(tuple(future_other_player), target))
        self.assertFalse(is_temporally_valid_peer(tuple(prior_same_player), target))


if __name__ == "__main__":
    unittest.main()
