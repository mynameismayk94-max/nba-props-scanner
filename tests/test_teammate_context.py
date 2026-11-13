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

    def test_wrong_team_is_not_read(self):
        summary = self._summary([
            {"athlete": {"id": "10"}, "stats": ["0:00", "--"]}
        ])
        self.assertEqual(extract_team_player_status(summary, "NYK"), {})


if __name__ == "__main__":
    unittest.main()
