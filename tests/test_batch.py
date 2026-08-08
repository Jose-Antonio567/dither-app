"""Testes dos utilitários de lote."""

from pathlib import Path
import tempfile
import unittest

from dither_app.batch import available_output_path, is_supported_image_path


class BatchTests(unittest.TestCase):
    def test_extensions_and_collisions(self) -> None:
        self.assertTrue(is_supported_image_path(Path("foto.PNG")))
        self.assertFalse(is_supported_image_path(Path("notas.txt")))
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = Path("foto.jpg")
            self.assertEqual(available_output_path(directory, source).name, "foto_dithered.png")
            (directory / "foto_dithered.png").touch()
            self.assertEqual(available_output_path(directory, source).name, "foto_dithered_2.png")
