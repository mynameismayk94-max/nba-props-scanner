import unittest
from unittest.mock import patch

from app.nba_injury_reports import (
    _status_for_player,
    _player_name_patterns,
    _normalize_matchup,
    get_official_nba_injury_status,
)


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

    def test_normalizes_espn_golden_state_abbreviation(self):
        self.assertEqual(_normalize_matchup("GS@MEM"), "GSW@MEM")
        report = "Matchup GSW@MEM Green, Draymond Out Injury/Illness"
        self.assertEqual(
            _status_for_player(report, "GS@MEM", "Draymond Green"),
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


    def test_searches_reports_published_more_than_two_hours_before_tipoff(self):
        report = (
            "Matchup WAS@TOR Toronto Raptors "
            "Barnes, Scottie Out Injury/Illness - Left Groin"
        )

        def fake_fetch(url):
            return report if "2026-11-18_01_30PM.pdf" in url else ""

        with patch("app.nba_injury_reports._fetch_report_text", side_effect=fake_fetch):
            result = get_official_nba_injury_status(
                "2026-11-18", "21:30", "WSH@TOR", "Scottie Barnes"
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "Out")
        self.assertIn("01_30PM", result["report_url"])

    def test_name_patterns_supports_full_name_and_report_order(self):
        patterns = _player_name_patterns("Draymond Green")
        self.assertEqual(len(patterns), 2)


if __name__ == "__main__":
    unittest.main()
