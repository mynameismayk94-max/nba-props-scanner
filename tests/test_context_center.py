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
