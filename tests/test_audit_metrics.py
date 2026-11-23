import unittest
from datetime import date, timedelta

from app.audit_metrics import build_gap_group_metrics, build_model_comparison, build_top10_vs_rest_metrics, is_temporally_valid_peer, spearman_correlation


def make_row(day, score=5.0, l5=4.0, l10=3.0, actual=4.0, games_last5=5, games_last10=10, audit_rank=1):
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
        games_last5,
        games_last10,
        audit_rank,
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

    def test_partial_windows_are_reported_and_excluded_from_full_window_comparison(self):
        rows = [
            make_row(0),
            make_row(1, games_last5=3, games_last10=3),
        ]
        result = build_model_comparison(rows)

        self.assertEqual(result["sample"], 2)
        self.assertEqual(result["full_window_sample"], 1)
        self.assertEqual(result["full_window_models"]["l10"]["sample"], 1)
        self.assertEqual(result["effective_sample_counts"]["l5"]["complete_window"], 1)
        self.assertEqual(result["effective_sample_counts"]["l5"]["partial_window"], 1)
        self.assertEqual(result["effective_sample_counts"]["l10"]["complete_window"], 1)
        self.assertEqual(result["effective_sample_counts"]["l10"]["partial_window"], 1)

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

    def test_top10_ranking_compares_player_adjusted_results_with_remaining_candidates(self):
        rows = [
            make_row(0, actual=5, l10=4, audit_rank=1),
            make_row(1, actual=7, l10=6, games_last10=5, audit_rank=2),
            make_row(0, actual=3, l10=4, audit_rank=11),
            make_row(1, actual=5, l10=6, games_last10=5, audit_rank=12),
        ]
        rows[2] = tuple(list(rows[2][:1]) + ["player-2"] + list(rows[2][2:]))
        rows[3] = tuple(list(rows[3][:1]) + ["player-2"] + list(rows[3][2:]))

        result = build_top10_vs_rest_metrics(rows, min_records=1, min_players=1)

        self.assertEqual(result["top10"]["records"], 2)
        self.assertEqual(result["rest"]["records"], 2)
        self.assertEqual(result["top10"]["l10_residual_sample"], 1)
        self.assertEqual(result["rest"]["l10_residual_sample"], 1)
        self.assertEqual(result["actual_average_difference"], 2.0)
        self.assertEqual(result["l10_residual_difference"], 2.0)
        self.assertEqual(result["distinct_dates"], 1)
        self.assertEqual(result["sample_status"], "collecting")

    def test_spearman_measures_monotonic_ranking_and_handles_ties(self):
        self.assertEqual(
            spearman_correlation([(1, 10), (2, 20), (3, 30)]),
            1.0
        )
        self.assertEqual(
            spearman_correlation([(1, 30), (2, 20), (3, 10)]),
            -1.0
        )
        self.assertEqual(
            spearman_correlation([(1, 10), (1, 20), (2, 30)]),
            0.866
        )
        self.assertIsNone(spearman_correlation([(1, 10), (1, 20)]))

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


def test_context_reference_metrics_compares_baseline_and_context():
    rows = [
        {
            "actual": 8,
            "player_l10_avg": 6,
            "opponent_position_minutes_filtered_defense_l10_avg": 7,
            "opponent_position_minute_band_defense_l10_avg": 8,
            "opponent_position_equal_game_defense_l10_avg": 5,
        },
        {
            "actual": 4,
            "player_l10_avg": 5,
            "opponent_position_minutes_filtered_defense_l10_avg": 4,
            "opponent_position_minute_band_defense_l10_avg": 6,
            "opponent_position_equal_game_defense_l10_avg": 5,
        },
    ]

    result = build_context_reference_metrics(rows)

    assert result["completed_records"] == 2
    filtered = result["references"]["opponent_position_minutes_filtered_defense_l10_avg"]
    assert filtered["sample"] == 2
    assert filtered["coverage_pct"] == 100.0
    assert filtered["baseline_l10_mae"] == 1.5
    assert filtered["context_mae"] == 0.5
    assert filtered["context_mae_delta_vs_l10"] == -1.0


def test_context_reference_metrics_excludes_missing_context_only():
    rows = [
        {
            "actual": 10,
            "player_l10_avg": 8,
            "opponent_position_minutes_filtered_defense_l10_avg": None,
            "opponent_position_minute_band_defense_l10_avg": 9,
            "opponent_position_equal_game_defense_l10_avg": 8,
        },
        {
            "actual": 5,
            "player_l10_avg": 5,
            "opponent_position_minutes_filtered_defense_l10_avg": 5,
            "opponent_position_minute_band_defense_l10_avg": None,
            "opponent_position_equal_game_defense_l10_avg": 5,
        },
    ]

    result = build_context_reference_metrics(rows)

    assert result["completed_records"] == 2
    assert result["references"]["opponent_position_minutes_filtered_defense_l10_avg"]["sample"] == 1
    assert result["references"]["opponent_position_minute_band_defense_l10_avg"]["sample"] == 1
