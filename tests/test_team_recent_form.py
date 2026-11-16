import unittest
from unittest.mock import patch

from app import main


def make_event(game_date, team_abbr, won, completed=True):
    own = {
        "team": {"abbreviation": team_abbr},
        "winner": won,
        "score": "110" if won else "95",
    }
    opponent = {
        "team": {"abbreviation": "OPP"},
        "winner": not won if completed else False,
        "score": "95" if won else "110",
    }
    return {
        "date": f"{game_date}T00:00Z",
        "status": {"type": {"completed": completed}},
        "competitions": [{
            "status": {"type": {"completed": completed}},
            "competitors": [own, opponent],
        }],
    }


class TeamRecentFormTests(unittest.TestCase):
    def setUp(self):
        main.fetch_team_schedule_cached.cache_clear()

    def tearDown(self):
        main.fetch_team_schedule_cached.cache_clear()

    @patch("app.main.request_json")
    def test_returns_last_five_completed_games_before_selected_date(self, request):
        request.return_value = {
            "events": [
                make_event("2026-11-01", "BOS", True),
                make_event("2026-11-03", "BOS", False),
                make_event("2026-11-05", "BOS", True),
                make_event("2026-11-07", "BOS", True),
                make_event("2026-11-09", "BOS", False),
                make_event("2026-11-11", "BOS", True),
                make_event("2026-11-13", "BOS", False),
                make_event("2026-11-16", "BOS", True),
                make_event("2026-11-17", "BOS", True),
                make_event("2026-11-14", "BOS", True, completed=False),
            ]
        }

        result = main.fetch_team_recent_form("BOS", "2026-11-16")

        self.assertEqual(result["last5"], ["W", "W", "L", "W", "L"])
        self.assertEqual(result["last5_record"], "3-2")
        self.assertEqual(result["last5_games"], 5)
        self.assertEqual(result["last5_status"], "ok")
        request.assert_called_once()
        self.assertEqual(
            request.call_args.args[0],
            main.TEAM_SCHEDULE_URL.format(team="bos")
        )
        self.assertEqual(request.call_args.args[1], {"season": 2027})

    @patch("app.main.request_json")
    def test_returns_unavailable_when_schedule_request_fails(self, request):
        request.side_effect = RuntimeError("ESPN indisponível")

        result = main.fetch_team_recent_form("BOS", "2026-11-16")

        self.assertEqual(result["last5"], [])
        self.assertIsNone(result["last5_record"])
        self.assertEqual(result["last5_games"], 0)
        self.assertEqual(result["last5_status"], "unavailable")

    @patch("app.main.request_json")
    def test_does_not_count_games_without_a_final_winner(self, request):
        request.return_value = {
            "events": [
                make_event("2026-11-10", "BOS", False, completed=False)
            ]
        }

        result = main.fetch_team_recent_form("BOS", "2026-11-16")

        self.assertEqual(result["last5"], [])
        self.assertEqual(result["last5_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
