"""Testes dos utilitários de lote."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox

from dither_app.batch import available_output_path, is_supported_image_path
from dither_app.main_window import MainWindow


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

    def test_list_management_and_thumbnails(self) -> None:
        application = QApplication.instance() or QApplication([])
        del application
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_directory = directory / "primeira"
            second_directory = directory / "segunda"
            first_directory.mkdir()
            second_directory.mkdir()
            first = first_directory / "igual.png"
            second = second_directory / "igual.png"
            third = directory / "terceira.png"
            for image_path, color in ((first, "red"), (second, "green"), (third, "blue")):
                Image.new("RGBA", (120, 80), color).save(image_path)

            window = MainWindow()
            try:
                window.add_images_from_paths([first, second, third])
                self.assertEqual(window.images_list.count(), 3)
                self.assertEqual(len(window.batch_items), 3)
                self.assertFalse(window.images_list.item(0).icon().isNull())

                window.images_list.setCurrentRow(1)
                window.remove_selected_batch_item()
                self.assertEqual(window.images_list.count(), 2)
                self.assertEqual(len(window.batch_items), 2)
                self.assertEqual([item.file_path for item in window.batch_items], [first, third])

                with patch.object(
                    QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
                ):
                    window.clear_batch_items()
                self.assertEqual(window.images_list.count(), 0)
                self.assertEqual(len(window.batch_items), 0)
                self.assertFalse(window.export_button.isEnabled())
                self.assertFalse(window.export_all_button.isEnabled())
                self.assertFalse(window.remove_image_button.isEnabled())
                self.assertFalse(window.clear_images_button.isEnabled())
            finally:
                window.close()
