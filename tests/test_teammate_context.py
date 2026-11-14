import unittest

from app.teammate_context import (
    compare_candidate_with_teammate,
    extract_team_player_status,
)


class TeammateContextTests(unittest.TestCase):
    def _summary(self, athletes):
        return {
            "boxscore": {
                "players": [{
                    "team": {"abbreviation": "BOS"},
                    "statistics": [{
                        "keys": ["minutes", "points"],
                        "athletes": athletes
                    }]
                }]
            }
        }

    def test_positive_minutes_means_played(self):
        result = extract_team_player_status(self._summary([
            {"athlete": {"id": "10"}, "stats": ["24:30", "12"]}
        ]), "BOS")
        self.assertEqual(result["10"]["status"], "played")

    def test_explicit_zero_minutes_means_confirmed_absence(self):
        result = extract_team_player_status(self._summary([
            {"athlete": {"id": "10"}, "stats": ["0:00", "--"]}
        ]), "BOS")
        self.assertEqual(result["10"]["status"], "confirmed_absent")
        self.assertEqual(result["10"]["evidence"], "explicit_zero_minutes")

    def test_missing_player_is_not_assumed_absent(self):
        result = extract_team_player_status(self._summary([]), "BOS")
        self.assertNotIn("10", result)

    def test_missing_minutes_is_unknown_unless_explicit_dnp(self):
        result = extract_team_player_status(self._summary([
            {"athlete": {"id": "10"}, "stats": ["--", "--"]}
        ]), "BOS")
        self.assertEqual(result["10"]["status"], "unknown")

    def test_candidate_split_excludes_unknown_and_requires_candidate_played(self):
        candidate_logs = [
            {"event_id": "g1", "minutes": 30, "points": 20},
            {"event_id": "g2", "minutes": 32, "points": 30},
            {"event_id": "g3", "minutes": 0, "points": 0},
        ]
        team_games = [{"game_id": "g1"}, {"game_id": "g2"}, {"game_id": "g3"}]
        statuses = {
            "g1": {"77": {"status": "played"}},
            "g2": {"77": {"status": "confirmed_absent"}},
            "g3": {"77": {"status": "confirmed_absent"}},
        }
        result = compare_candidate_with_teammate(
            candidate_logs, team_games, statuses, "77", "points"
        )
        self.assertEqual(result["with_teammate"]["average"], 20)
        self.assertEqual(result["confirmed_absent"]["average"], 30)
        self.assertEqual(result["difference_absent_minus_played"], 10)


    def test_shortlist_requires_team_presence_frequency_and_minutes(self):
        roster = [
            {"id": "candidate", "player": "Candidate"},
            {"id": "star", "player": "Star", "position": "G"},
            {"id": "traded", "player": "Former teammate"},
            {"id": "bench", "player": "Bench"},
        ]
        game_ids = [f"g{i}" for i in range(10)]
        player_logs = {
            "star": ({"player": "Star", "position": "G"}, tuple(
                {"event_id": game_ids[i], "team": "BOS", "minutes": 30,
                 "points": 20, "assists": 5, "fga": 15}
                for i in range(8)
            )),
            "traded": ({"player": "Former teammate"}, tuple(
                {"event_id": game_ids[i], "team": "NYK", "minutes": 35,
                 "points": 25, "assists": 7, "fga": 18}
                for i in range(10)
            )),
            "bench": ({"player": "Bench"}, tuple(
                {"event_id": game_ids[i], "team": "BOS", "minutes": 12,
                 "points": 4, "assists": 1, "fga": 3}
                for i in range(10)
            )),
        }
        from app.teammate_context import build_teammate_shortlist
        shortlist = build_teammate_shortlist(
            "candidate", "BOS", roster, player_logs,
            [{"game_id": game_id} for game_id in game_ids]
        )
        self.assertEqual([item["player_id"] for item in shortlist], ["star"])
        self.assertEqual(shortlist[0]["games_in_window"], 8)

    def test_shortlist_caps_at_three(self):
        roster = [{"id": f"p{i}", "player": f"Player {i}"} for i in range(5)]
        logs = {
            f"p{i}": ({"player": f"Player {i}"}, tuple(
                {"event_id": f"g{j}", "team": "BOS", "minutes": 28-i,
                 "points": 15, "assists": 3, "fga": 10}
                for j in range(7)
            ))
            for i in range(5)
        }
        from app.teammate_context import build_teammate_shortlist
        shortlist = build_teammate_shortlist(
            "p0", "BOS", roster, logs,
            [{"game_id": f"g{j}"} for j in range(7)], limit=10
        )
        self.assertLessEqual(len(shortlist), 3)
        self.assertNotIn("p0", [item["player_id"] for item in shortlist])


    def test_later_stat_block_does_not_overwrite_confirmed_minutes(self):
        summary = {
            "boxscore": {
                "players": [{
                    "team": {"abbreviation": "BOS"},
                    "statistics": [
                        {"keys": ["MIN"], "athletes": [
                            {"athlete": {"id": "10"}, "stats": ["28:00"]}
                        ]},
                        {"keys": ["PTS", "AST"], "athletes": [
                            {"athlete": {"id": "10"}, "stats": ["18", "4"]}
                        ]}
                    ]
                }]
            }
        }
        result = extract_team_player_status(summary, "BOS")
        self.assertEqual(result["10"]["status"], "played")
        self.assertEqual(result["10"]["minutes"], 28)


    def test_full_context_compares_all_metrics_and_excludes_unknown(self):
        from app.teammate_context import build_candidate_teammate_context

        game_ids = [f"g{i}" for i in range(1, 7)]
        point_values = [20, 30, 25, 40, 22, 24]
        candidate_logs = [
            {"event_id": game_id, "team": "BOS", "minutes": 30,
             "points": points, "assists": 5, "rebounds": 7,
             "three_pa": 4, "two_pa": 8, "fgm": 8}
            for game_id, points in zip(game_ids, point_values)
        ]
        teammate_logs = tuple(
            {"event_id": game_id, "team": "BOS", "minutes": 30,
             "points": 18, "assists": 4, "fga": 14}
            for game_id in ("g1", "g2", "g4", "g5", "g6")
        )
        result = build_candidate_teammate_context(
            candidate_id="candidate",
            team="BOS",
            roster=[
                {"id": "candidate", "player": "Candidate"},
                {"id": "mate", "player": "Important teammate"},
            ],
            player_logs_by_id={
                "candidate": ({"player": "Candidate"}, tuple(candidate_logs)),
                "mate": ({"player": "Important teammate"}, teammate_logs),
            },
            team_games=[{"game_id": game_id} for game_id in game_ids],
            teammate_status_by_game={
                "g1": {"mate": {"status": "played"}},
                "g2": {"mate": {"status": "confirmed_absent"}},
                "g3": {"mate": {"status": "unknown"}},
                "g4": {"mate": {"status": "confirmed_absent"}},
                "g5": {"mate": {"status": "played"}},
                "g6": {"mate": {"status": "played"}},
            }
        )
        comparison = result["comparisons"]["mate"]["points"]
        self.assertEqual(comparison["with_teammate"]["games"], 3)
        self.assertEqual(comparison["confirmed_absent"]["games"], 2)
        self.assertEqual(comparison["unknown_games_excluded"], 1)
        self.assertEqual(comparison["difference_absent_minus_played"], 13)
        self.assertIn("assists", result["comparisons"]["mate"])
        self.assertIn("minutes", result["comparisons"]["mate"])

    def test_wrong_team_is_not_read(self):
        summary = self._summary([
            {"athlete": {"id": "10"}, "stats": ["0:00", "--"]}
        ])
        self.assertEqual(extract_team_player_status(summary, "NYK"), {})


if __name__ == "__main__":
    unittest.main()
