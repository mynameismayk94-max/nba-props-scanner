import unittest
from datetime import date

from app.main import build_player_venue_production


class PlayerVenueProductionTests(unittest.TestCase):
    def test_home_and_away_are_calculated_independently(self):
        games = [
            {"date": "2026-11-12", "home_away": "home", "points": 20, "minutes": 30},
            {"date": "2026-11-11", "home_away": "away", "points": 10, "minutes": 28},
            {"date": "2026-11-10", "home_away": "home", "points": 30, "minutes": 32},
            {"date": "2026-11-09", "home_away": "away", "points": 14, "minutes": 26},
            {"date": "2026-11-08", "home_away": None, "points": 99, "minutes": 35},
            {"date": "2026-11-14", "home_away": "home", "points": 100, "minutes": 40},
        ]

        result = build_player_venue_production(
            games, date(2026, 11, 13), "points"
        )

        self.assertEqual(result["home"]["season"]["games"], 2)
        self.assertEqual(result["home"]["season"]["average"], 25.0)
        self.assertEqual(result["away"]["season"]["games"], 2)
        self.assertEqual(result["away"]["season"]["average"], 12.0)
        self.assertEqual(result["home"]["last10"]["minutes_avg"], 31.0)
        self.assertEqual(result["away"]["last10"]["minutes_avg"], 27.0)

    def test_empty_venue_returns_null_average_and_zero_games(self):
        result = build_player_venue_production(
            [{"date": "2026-11-12", "home_away": None, "points": 20, "minutes": 30}],
            date(2026, 11, 13),
            "points"
        )

        for venue in ("home", "away"):
            self.assertEqual(result[venue]["season"]["games"], 0)
            self.assertIsNone(result[venue]["season"]["average"])
            self.assertEqual(result[venue]["last10"]["games"], 0)


if __name__ == "__main__":
    unittest.main()
