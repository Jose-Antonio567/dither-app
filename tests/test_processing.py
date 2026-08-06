"""Testes básicos dos processamentos de imagem."""

import unittest

import numpy as np
from PIL import Image

from dither_app.processing import process_image_with_settings, remove_background_by_color
from dither_app.settings import ProcessingSettings


class ProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        pixels = np.array(
            [[[255, 255, 255, 255], [20, 30, 40, 255]],
             [[80, 90, 100, 128], [200, 30, 90, 0]]],
            dtype=np.uint8,
        )
        self.image = Image.fromarray(pixels, "RGBA")
        self.original_pixels = np.array(self.image).copy()

    def test_methods_keep_rgba_and_size(self) -> None:
        methods = ["Nenhum", "Threshold", "Floyd–Steinberg", "Atkinson",
                   "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Ruído"]
        for method in methods:
            with self.subTest(method=method):
                settings = ProcessingSettings(method, 4, 128, 50, True, False)
                result = process_image_with_settings(self.image, settings)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, self.image.size)
                self.assertTrue(np.array_equal(np.array(self.image), self.original_pixels))

    def test_background_connected_only(self) -> None:
        pixels = np.full((5, 5, 4), 255, dtype=np.uint8)
        pixels[1:4, 1:4, :3] = 0
        pixels[2, 2, :3] = 255
        image = Image.fromarray(pixels, "RGBA")
        connected = np.array(remove_background_by_color(image, (255, 255, 255), 0, True))
        global_result = np.array(remove_background_by_color(image, (255, 255, 255), 0, False))
        self.assertEqual(connected[0, 0, 3], 0)
        self.assertEqual(connected[2, 2, 3], 255)
        self.assertEqual(global_result[2, 2, 3], 0)


if __name__ == "__main__":
    unittest.main()
