"""Testes de persistência de predefinições."""

from pathlib import Path
import tempfile
import unittest

from dither_app.presets import PresetManager, UserPreset, settings_from_dict, settings_to_dict, validate_preset_name
from dither_app.settings import ProcessingSettings


class PresetTests(unittest.TestCase):
    def test_round_trip_and_atomic_save(self) -> None:
        settings = ProcessingSettings("Bayer 4x4", 8, 120, 40, True, False, pixel_scale=4, background_color=(10, 20, 30))
        self.assertEqual(settings_from_dict(settings_to_dict(settings)), settings)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "presets.json"
            manager = PresetManager(path)
            manager.save([UserPreset("Retrô ágil", settings)])
            self.assertTrue(path.is_file())
            self.assertFalse((path.parent / "presets.tmp.json").exists())
            self.assertEqual(manager.load()[0].settings.background_color, (10, 20, 30))

    def test_invalid_files_and_names(self) -> None:
        with self.assertRaises(ValueError): validate_preset_name("normal")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "presets.json"
            manager = PresetManager(path)
            self.assertEqual(manager.load(), [])
            path.write_text("{inválido", encoding="utf-8")
            self.assertEqual(manager.load(), [])
            path.write_text('{"schema_version": 2, "presets": []}', encoding="utf-8")
            self.assertEqual(manager.load(), [])
            self.assertTrue(manager.incompatible_schema)

    def test_duplicate_names_are_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = PresetManager(Path(temporary_directory) / "presets.json")
            settings = ProcessingSettings("Nenhum", 16, 128, 50, True, False)
            with self.assertRaises(ValueError):
                manager.save([UserPreset("Pixel Art", settings), UserPreset("pixel art", settings)])
