import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.main import app, require_admin_api_key


class AdminWriteRouteSecurityTests(unittest.TestCase):
    def test_missing_admin_key_configuration_fails_closed(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADMIN_API_KEY", None)
            with self.assertRaises(HTTPException) as context:
                require_admin_api_key(None)
        self.assertEqual(context.exception.status_code, 503)

    def test_missing_or_invalid_key_is_rejected(self):
        with patch.dict(os.environ, {"ADMIN_API_KEY": "expected-secret"}):
            with self.assertRaises(HTTPException) as missing:
                require_admin_api_key(None)
            with self.assertRaises(HTTPException) as invalid:
                require_admin_api_key("wrong-secret")
        self.assertEqual(missing.exception.status_code, 401)
        self.assertEqual(invalid.exception.status_code, 401)

    def test_valid_key_is_accepted(self):
        with patch.dict(os.environ, {"ADMIN_API_KEY": "expected-secret"}):
            self.assertTrue(require_admin_api_key("expected-secret"))

    def test_write_routes_require_admin_dependency(self):
        paths = app.openapi()["paths"]
        for path, method in (
            ("/capture-audit-snapshot", "get"),
            ("/refresh-audit", "get"),
            ("/collect-results", "get"),
            ("/collect-pending-results", "get"),
            ("/context/defense/refresh", "post"),
        ):
            operation = paths[path][method]
            self.assertTrue(
                operation.get("security"),
                f"{method.upper()} {path} should declare admin security"
            )


if __name__ == "__main__":
    unittest.main()
