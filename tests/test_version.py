"""Testes dos metadados públicos da versão."""

import unittest

from dither_app.version import APP_VERSION, APP_VERSION_LABEL


class VersionTests(unittest.TestCase):
    """Garante a versão entregue pela release atual."""

    def test_public_version(self) -> None:
        self.assertEqual(APP_VERSION, "0.11.0")
        self.assertEqual(APP_VERSION_LABEL, "0.11.0 Beta")


if __name__ == "__main__":
    unittest.main()
