import unittest
from contextlib import contextmanager
from datetime import date

from app.context_center import create_context_router


class RecordingCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        self.connection.statements.append(("execute", query, params))

    def executemany(self, query, rows):
        self.connection.statements.append(("executemany", query, list(rows)))


class RecordingConnection:
    def __init__(self, statements):
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return RecordingCursor(self)


class ContextRefreshTests(unittest.TestCase):
    def run_refresh(self, histories):
        statements = []

        @contextmanager
        def connection_factory():
            yield RecordingConnection(statements)

        def history_provider(team, limit, target_date):
            value = histories[team]
            if isinstance(value, Exception):
                raise value
            return value

        router = create_context_router(
            get_database_connection=connection_factory,
            get_team_defense_history=history_provider,
            team_abbreviations=("AAA", "BBB", "CCC"),
        )
        route = next(
            item for item in router.routes
            if getattr(item, "path", None) == "/context/defense/refresh"
        )
        response = route.endpoint(selected_date="2026-11-13")
        return response, statements

    def test_complete_refresh_persists_all_teams(self):
        history = {"games_analyzed": 10}
        response, statements = self.run_refresh({
            "AAA": history, "BBB": history, "CCC": history
        })
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["teams_collected"], 3)
        self.assertEqual(response["teams_saved"], 3)
        writes = [item for item in statements if item[0] == "executemany"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(writes[0][2]), 3)

    def test_partial_refresh_does_not_persist_mixed_snapshot(self):
        history = {"games_analyzed": 10}
        response, statements = self.run_refresh({
            "AAA": history, "BBB": RuntimeError("ESPN indisponível"),
            "CCC": history
        })
        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["teams_collected"], 2)
        self.assertEqual(response["teams_saved"], 0)
        self.assertEqual(response["teams_with_errors"], 1)
        self.assertFalse(any(item[0] == "executemany" for item in statements))

    def test_total_failure_does_not_persist(self):
        response, statements = self.run_refresh({
            "AAA": RuntimeError("erro A"),
            "BBB": RuntimeError("erro B"),
            "CCC": None,
        })
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["teams_collected"], 0)
        self.assertEqual(response["teams_saved"], 0)
        self.assertFalse(any(item[0] == "executemany" for item in statements))


if __name__ == "__main__":
    unittest.main()


class ContextReadTests(unittest.TestCase):
    def setUp(self):
        from datetime import datetime, timezone

        self.teams = tuple(f"TM{i:02d}" for i in range(30))
        self.captured_at = datetime(2026, 11, 13, tzinfo=timezone.utc)
        self.rows = []
        for index, team in enumerate(self.teams):
            positions = {}
            for position in ("G", "F", "C"):
                positions[position] = {
                    "averages": {
                        stat: float(index + stat_index + 1)
                        for stat_index, stat in enumerate(
                            ("points", "assists", "rebounds", "three_pa", "two_pa", "fgm")
                        )
                    },
                    "stat_games_analyzed": {
                        stat: 5 if stat != "fgm" else 4
                        for stat in ("points", "assists", "rebounds", "three_pa", "two_pa", "fgm")
                    },
                    "players_per_game_avg": 4.0
                }
            payload = {
                "L5": {"positions": positions},
                "L10": {"positions": positions},
                "venue_windows": {
                    "home": {
                        "L5": {"positions": positions},
                        "L10": {"positions": positions}
                    },
                    "away": {
                        "L5": {"positions": positions},
                        "L10": {"positions": positions}
                    }
                }
            }
            self.rows.append((team, payload, self.captured_at))

        @contextmanager
        def connection_factory():
            yield self

        self.statements = []
        self.cursor_factory = connection_factory
        self.router = create_context_router(
            get_database_connection=connection_factory,
            get_team_defense_history=lambda *_: {},
            team_abbreviations=self.teams
        )
        self.ranking_route = next(
            route for route in self.router.routes
            if getattr(route, "path", None) == "/context/defense/rankings"
        )
        self.detail_route = next(
            route for route in self.router.routes
            if getattr(route, "path", None) == "/context/defense/team/{team}"
        )

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        self.statements.append((query, params))

    def fetchall(self):
        return self.rows

    def test_all_position_stat_window_combinations_return_complete_sorted_rankings(self):
        stats = ("points", "assists", "rebounds", "three_pa", "two_pa", "fgm")
        for position in ("G", "F", "C"):
            for stat in stats:
                for window in ("L5", "L10"):
                    with self.subTest(position=position, stat=stat, window=window):
                        result = self.ranking_route.endpoint(
                            selected_date="2026-11-13",
                            position=position,
                            stat=stat,
                            window=window
                        )
                        self.assertEqual(result["status"], "ok")
                        self.assertEqual(result["teams_available"], 30)
                        self.assertEqual(len(result["ranking"]), 30)
                        ranked_teams = [item["team"] for item in result["ranking"]]
                        self.assertEqual(len(ranked_teams), len(set(ranked_teams)))
                        values = [item["allowed_per_game"] for item in result["ranking"]]
                        self.assertEqual(values, sorted(values))
                        self.assertEqual(result["ranking"][0]["games_analyzed"], 4 if stat == "fgm" else 5)

    def test_team_detail_returns_saved_history_for_all_positions(self):
        result = self.detail_route.endpoint(team="TM00", selected_date="2026-11-13")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["team"], "TM00")
        self.assertIn("L5", result["history"])
        self.assertIn("L10", result["history"])
        for position in ("G", "F", "C"):
            self.assertEqual(
                set(result["history"]["L5"]["positions"][position]["averages"]),
                {"points", "assists", "rebounds", "three_pa", "two_pa", "fgm"}
            )

    def test_invalid_ranking_filters_are_rejected(self):
        from fastapi import HTTPException

        invalid_cases = (
            {"position": "OTHER", "stat": "points", "window": "L5"},
            {"position": "G", "stat": "fga", "window": "L5"},
            {"position": "G", "stat": "points", "window": "L20"}
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(HTTPException):
                    self.ranking_route.endpoint(
                        selected_date="2026-11-13",
                        **case
                    )


    def test_home_and_away_rankings_read_their_own_persisted_windows(self):
        for venue in ("home", "away"):
            with self.subTest(venue=venue):
                result = self.ranking_route.endpoint(
                    selected_date="2026-11-13",
                    position="G",
                    stat="points",
                    window="L5",
                    venue=venue
                )
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["venue"], venue)
                self.assertEqual(result["teams_available"], 30)
                self.assertEqual(len(result["ranking"]), 30)

    def test_invalid_venue_is_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            self.ranking_route.endpoint(
                selected_date="2026-11-13",
                position="G",
                stat="points",
                window="L5",
                venue="neutral"
            )
