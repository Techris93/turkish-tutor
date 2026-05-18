import unittest

from oauth_flow import OAuthError, fetch_google_profile


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload

    async def get(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeResponse(self.payload)


class OAuthFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_google_profile_requires_verified_email(self):
        with self.assertRaises(OAuthError):
            await fetch_google_profile(
                FakeClient({"email": "unverified@example.com", "email_verified": False}),
                "token",
            )

        profile = await fetch_google_profile(
            FakeClient({"email": "verified@example.com", "email_verified": True, "name": "Verified", "sub": "123"}),
            "token",
        )
        self.assertEqual(profile.email, "verified@example.com")


if __name__ == "__main__":
    unittest.main()
