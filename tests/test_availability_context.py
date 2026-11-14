import unittest

from app.availability_context import (
    classify_injury_status,
    parse_espn_injuries,
    select_eligible_players,
)


class AvailabilityContextTests(unittest.TestCase):
    def test_out_is_confirmed_absence(self):
        self.assertEqual(classify_injury_status("Out"), "confirmed_out")

    def test_questionable_is_not_confirmed_absence(self):
        self.assertEqual(classify_injury_status("Questionable"), "uncertain")

    def test_day_to_day_is_not_confirmed_absence(self):
        self.assertEqual(classify_injury_status("Day-To-Day"), "uncertain")

    def test_parse_league_report(self):
        payload = {
            "injuries": [{
                "team": {"abbreviation": "BOS"},
                "injuries": [{
                    "athlete": {"id": "10", "displayName": "Player A"},
                    "status": "Out",
                    "type": {"name": "Knee"},
                    "details": "soreness",
                }]
            }]
        }
        result = parse_espn_injuries(payload)
        self.assertEqual(result["10"]["availability"], "confirmed_out")
        self.assertEqual(result["10"]["team"], "BOS")

    def test_selection_removes_only_confirmed_out_and_fills_to_ten(self):
        results = [
            {"player_id": str(i), "player": f"P{i}", "triage_score": 10-i, "team": "BOS"}
            for i in range(1, 13)
        ]
        injury_index = {
            "2": {"availability": "confirmed_out", "status_raw": "Out"},
            "5": {"availability": "uncertain", "status_raw": "Questionable"},
            "7": {"availability": "confirmed_out", "status_raw": "Inactive"},
        }
        eligible, removed = select_eligible_players(results, injury_index, limit=10)
        self.assertEqual(len(eligible), 10)
        self.assertEqual(
            [p["player_id"] for p in eligible],
            ["1", "3", "4", "5", "6", "8", "9", "10", "11", "12"]
        )
        self.assertEqual([p["player_id"] for p in removed], ["2", "7"])


if __name__ == "__main__":
    unittest.main()
