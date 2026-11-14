import unittest
from app.availability_context import classify_injury_status, parse_espn_injuries, select_eligible_players

class AvailabilityContextTests(unittest.TestCase):
    def test_out_and_inactive_are_confirmed_absences(self):
        self.assertEqual(classify_injury_status("Out"), "confirmed_out")
        self.assertEqual(classify_injury_status("Inactive"), "confirmed_out")

    def test_uncertain_statuses_are_not_confirmed_absences(self):
        self.assertEqual(classify_injury_status("Questionable"), "uncertain")
        self.assertEqual(classify_injury_status("Day-To-Day"), "uncertain")

    def test_parse_league_report(self):
        payload = {"injuries": [{"team": {"abbreviation": "BOS"}, "injuries": [{
            "athlete": {"id": "10", "displayName": "Player A"},
            "status": "Out", "type": {"name": "Knee"}, "details": "soreness"
        }]}]}
        result = parse_espn_injuries(payload)
        self.assertEqual(result["10"]["availability"], "confirmed_out")
        self.assertEqual(result["10"]["team"], "BOS")

    def test_only_initial_top_ten_outs_are_reported_and_next_eligible_moves_up(self):
        results = [
            {"player_id": str(i), "player": f"P{i}", "triage_score": 100-i, "team": "BOS"}
            for i in range(1, 16)
        ]
        injury_index = {
            "2": {"availability": "confirmed_out", "status_raw": "Out"},
            "5": {"availability": "uncertain", "status_raw": "Questionable"},
            "7": {"availability": "confirmed_out", "status_raw": "Inactive"},
            "12": {"availability": "confirmed_out", "status_raw": "Out"},
        }
        displayed, removed_top, outside_top, eligible_all = select_eligible_players(results, injury_index, 10)
        self.assertEqual([p["player_id"] for p in displayed], ["1","3","4","5","6","8","9","10","11","13"])
        self.assertEqual([p["player_id"] for p in removed_top], ["2","7"])
        self.assertEqual([p["player_id"] for p in outside_top], ["14","15"])
        self.assertEqual(len(eligible_all), 12)
        self.assertEqual(displayed[3]["availability"]["availability"], "uncertain")
        self.assertNotIn("12", [p["player_id"] for p in removed_top])
        self.assertNotIn("12", [p["player_id"] for p in outside_top])

    def test_no_injury_report_does_not_mean_out(self):
        displayed, removed, outside, all_eligible = select_eligible_players(
            [{"player_id":"1","player":"P1","team":"BOS"}], {}, 10
        )
        self.assertEqual(displayed[0]["availability"]["availability"], "no_injury_report")
        self.assertEqual(removed, [])
        self.assertEqual(outside, [])
        self.assertEqual(len(all_eligible), 1)

if __name__ == "__main__":
    unittest.main()
