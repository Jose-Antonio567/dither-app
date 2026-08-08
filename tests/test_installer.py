"""Validação estática da configuração Inno Setup."""

from pathlib import Path
import unittest


class InstallerConfigurationTests(unittest.TestCase):
    def test_inno_setup_configuration(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        script = (project_root / "installer" / "DitherApp.iss").read_text(encoding="utf-8")
        self.assertIn("64850F7E-7186-46EC-9E33-4D6D91D7EB26", script)
        self.assertIn("DitherApp.exe", script)
        self.assertIn("OutputBaseFilename=DitherApp-Setup-0.10.0-beta-win64", script)
        self.assertNotIn("C:\\Users", script)
        self.assertTrue((project_root / "build-installer.ps1").is_file())
