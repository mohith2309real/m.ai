"""Regression tests for m.ai's local and network inference authentication."""
import os
import unittest
from unittest.mock import Mock, patch

import mai_serve


class AuthenticationTests(unittest.TestCase):
    def authed(self, host, token=None, supplied=""):
        handler = Mock()
        handler.server.server_address = (host, 8123)
        handler.headers.get.return_value = supplied
        with patch.dict(os.environ, {"MAI_API_KEY": token} if token is not None else {}, clear=True):
            return mai_serve.Handler._authed(handler)

    def test_loopback_defaults_to_local_only(self):
        self.assertTrue(self.authed("127.0.0.1"))
        self.assertTrue(self.authed("::1"))
        self.assertFalse(self.authed("0.0.0.0"))
        self.assertFalse(self.authed("192.168.1.20"))

    def test_public_bind_requires_exact_bearer_token(self):
        self.assertTrue(self.authed("0.0.0.0", "secret", "Bearer secret"))
        self.assertFalse(self.authed("0.0.0.0", "secret", "Bearer wrong"))
        self.assertFalse(self.authed("0.0.0.0", "secret", ""))


if __name__ == "__main__":
    unittest.main()
