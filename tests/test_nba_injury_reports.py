import unittest

from app.nba_injury_reports import _status_for_player, _player_name_patterns


class OfficialNbaInjuryReportTests(unittest.TestCase):
    def test_matches_official_last_first_name_and_out_status(self):
        report = (
            "Matchup GSW@MEM Golden State Warriors "
            "Green, Draymond Out Injury/Illness - Right Elbow Injury Management"
        )
        self.assertEqual(
            _status_for_player(report, "GSW@MEM", "Draymond Green"),
            "Out",
        )

    def test_does_not_infer_dnp_from_missing_player(self):
        report = "Matchup GSW@MEM Golden State Warriors Curry, Stephen Out"
        self.assertIsNone(
            _status_for_player(report, "GSW@MEM", "Draymond Green")
        )

    def test_requires_requested_matchup(self):
        report = (
            "Matchup LAL@GSW Golden State Warriors "
            "Green, Draymond Out Injury/Illness - Right Elbow"
        )
        self.assertIsNone(
            _status_for_player(report, "GSW@MEM", "Draymond Green")
        )

    def test_name_patterns_supports_full_name_and_report_order(self):
        patterns = _player_name_patterns("Draymond Green")
        self.assertEqual(len(patterns), 2)


if __name__ == "__main__":
    unittest.main()
