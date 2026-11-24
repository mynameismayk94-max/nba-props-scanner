import asyncio
import unittest
from unittest.mock import patch

from app.main import _rate_limit_buckets, rate_limit_requests


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        _rate_limit_buckets.clear()

    def tearDown(self):
        _rate_limit_buckets.clear()

    def test_candidates_are_rejected_after_configured_limit(self):
        class Client:
            host = "198.51.100.10"

        class URL:
            path = "/candidates"

        class Request:
            client = Client()
            url = URL()
            headers = {}

        async def next_handler(request):
            return "ok"

        async def run():
            with patch("app.main._RATE_LIMIT_WINDOW_SECONDS", 60), patch(
                "app.main._RATE_LIMIT_MAX_REQUESTS", 2
            ):
                first = await rate_limit_requests(Request(), next_handler)
                second = await rate_limit_requests(Request(), next_handler)
                third = await rate_limit_requests(Request(), next_handler)
                return first, second, third

        first, second, third = asyncio.run(run())
        self.assertEqual(first, "ok")
        self.assertEqual(second, "ok")
        self.assertEqual(third.status_code, 429)

    def test_audit_reads_are_not_rate_limited(self):
        class Client:
            host = "198.51.100.12"

        class URL:
            path = "/validation-history"

        class Request:
            client = Client()
            url = URL()
            headers = {}

        async def next_handler(request):
            return "ok"

        async def run():
            with patch("app.main._RATE_LIMIT_MAX_REQUESTS", 1):
                responses = [
                    await rate_limit_requests(Request(), next_handler)
                    for _ in range(3)
                ]
                return responses

        self.assertEqual(asyncio.run(run()), ["ok", "ok", "ok"])

    def test_health_is_exempt(self):
        class Client:
            host = "198.51.100.11"

        class URL:
            path = "/health"

        class Request:
            client = Client()
            url = URL()
            headers = {}

        async def next_handler(request):
            return "ok"

        async def run():
            with patch("app.main._RATE_LIMIT_MAX_REQUESTS", 1):
                await rate_limit_requests(Request(), next_handler)
                return await rate_limit_requests(Request(), next_handler)

        self.assertEqual(asyncio.run(run()), "ok")


if __name__ == "__main__":
    unittest.main()
