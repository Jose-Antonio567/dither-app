"""Interface inicial do Dither App com importação e pré-visualização."""

import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PySide6.QtCore import QObject, QRect, QSignalBlocker, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ProcessingSettings:
    """Configurações capturadas para uma prévia ou exportação."""

    method: str
    color_count: int
    threshold: int
    noise_intensity: int
    preserve_transparency: bool
    hard_alpha: bool
    noise_seed: int = 12345


def apply_alpha_settings(
    image: Image.Image,
    preserve_transparency: bool,
    hard_alpha: bool,
) -> Image.Image:
    """Aplica somente as opções de alfa a uma cópia RGBA."""
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    pixels[:, :, 3] = calculate_final_alpha(
        pixels[:, :, 3], preserve_transparency, hard_alpha
    )
    pixels[pixels[:, :, 3] == 0, :3] = 0
    return Image.fromarray(pixels, mode="RGBA")


def process_image_with_settings(
    image: Image.Image,
    settings: ProcessingSettings,
    cancel_callback: Callable[[], bool] | None = None,
) -> Image.Image:
    """Processa uma imagem usando um conjunto fechado de configurações."""
    method = settings.method
    if method == "Nenhum":
        return apply_alpha_settings(
            image, settings.preserve_transparency, settings.hard_alpha
        )
    if method == "Threshold":
        return apply_threshold(
            image, settings.threshold, settings.preserve_transparency, settings.hard_alpha
        )
    if method == "Floyd–Steinberg":
        return apply_floyd_steinberg(
            image, settings.color_count, settings.preserve_transparency, settings.hard_alpha
        )
    if method == "Atkinson":
        return apply_atkinson(
            image, settings.color_count, settings.preserve_transparency,
            settings.hard_alpha, cancel_callback
        )
    if method.startswith("Bayer "):
        matrix_size = int(method.split()[1].split("x")[0])
        return apply_bayer_dithering(
            image, settings.color_count, matrix_size, settings.preserve_transparency,
            settings.hard_alpha, cancel_callback
        )
    if method == "Ruído":
        return apply_noise_dithering(
            image, settings.color_count, settings.noise_intensity,
            settings.preserve_transparency, settings.hard_alpha, settings.noise_seed,
            cancel_callback
        )
    raise ValueError(f"Método de dithering desconhecido: {method}")


def apply_threshold(
    image: Image.Image,
    threshold: int,
    preserve_transparency: bool,
    hard_alpha: bool,
) -> Image.Image:
    """Cria uma imagem preta e branca a partir de uma imagem RGBA."""
    source_pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    result_pixels = np.empty_like(source_pixels)

    # Calcula luminância sem percorrer os pixels manualmente.
    rgb_pixels = source_pixels[:, :, :3].astype(np.float32)
    luminance = (
        0.2126 * rgb_pixels[:, :, 0]
        + 0.7152 * rgb_pixels[:, :, 1]
        + 0.0722 * rgb_pixels[:, :, 2]
    )
    black_or_white = np.where(luminance >= threshold, 255, 0).astype(np.uint8)
    result_pixels[:, :, :3] = black_or_white[:, :, np.newaxis]

    result_pixels[:, :, 3] = calculate_final_alpha(
        source_pixels[:, :, 3], preserve_transparency, hard_alpha
    )

    # Evita cores invisíveis em pixels totalmente transparentes.
    result_pixels[result_pixels[:, :, 3] == 0, :3] = 0
    return Image.fromarray(result_pixels, mode="RGBA")


def calculate_final_alpha(
    original_alpha: np.ndarray,
    preserve_transparency: bool,
    hard_alpha: bool,
) -> np.ndarray:
    """Calcula o canal alfa final conforme as opções escolhidas."""
    if not preserve_transparency:
        return np.full_like(original_alpha, 255)

    final_alpha = original_alpha.copy()
    if hard_alpha:
        final_alpha = np.where(final_alpha >= 128, 255, 0).astype(np.uint8)
    return final_alpha


def create_adaptive_palette(
    source_pixels: np.ndarray,
    final_alpha: np.ndarray,
    color_count: int,
) -> tuple[Image.Image | None, np.ndarray]:
    """Cria uma paleta P usando somente pixels que permanecerão visíveis."""
    visible_pixels = source_pixels[final_alpha > 0, :3]
    if visible_pixels.size == 0:
        return None, np.empty((0, 3), dtype=np.uint8)

    # Limita a amostra mantendo intervalos regulares e determinísticos.
    if len(visible_pixels) > 100_000:
        sample_indices = np.linspace(
            0, len(visible_pixels) - 1, num=100_000, dtype=np.intp
        )
        visible_pixels = visible_pixels[sample_indices]

    palette_sample = Image.fromarray(visible_pixels.reshape(-1, 1, 3), mode="RGB")
    palette_image = palette_sample.quantize(
        colors=color_count,
        method=Image.Quantize.MEDIANCUT,
    )
    palette_values = np.asarray(palette_image.getpalette(), dtype=np.uint8).reshape(-1, 3)
    used_indices = np.unique(np.asarray(palette_image))
    return palette_image, palette_values[used_indices]


def create_color_lut(palette_colors: np.ndarray) -> np.ndarray:
    """Monta uma LUT RGB de 5 bits para a cor mais próxima da paleta."""
    color_levels = np.linspace(0, 255, num=32, dtype=np.float32)
    red, green, blue = np.meshgrid(
        color_levels, color_levels, color_levels, indexing="ij"
    )
    lut_samples = np.stack((red, green, blue), axis=-1).reshape(-1, 3)
    color_distances = np.sum(
        (lut_samples[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    nearest_indices = np.argmin(color_distances, axis=1)
    return palette_colors[nearest_indices].reshape(32, 32, 32, 3)


def apply_floyd_steinberg(
    image: Image.Image,
    color_count: int,
    preserve_transparency: bool,
    hard_alpha: bool,
) -> Image.Image:
    """Aplica dithering Floyd–Steinberg colorido com paleta adaptativa."""
    color_count = max(2, min(color_count, 256))
    rgba_image = image.convert("RGBA")
    source_pixels = np.asarray(rgba_image, dtype=np.uint8)
    final_alpha = calculate_final_alpha(
        source_pixels[:, :, 3], preserve_transparency, hard_alpha
    )
    palette_image, _ = create_adaptive_palette(
        source_pixels, final_alpha, color_count
    )
    if palette_image is None:
        return Image.new("RGBA", rgba_image.size, (0, 0, 0, 0))
    quantized_image = rgba_image.convert("RGB").quantize(
        colors=color_count,
        dither=Image.Dither.FLOYDSTEINBERG,
        palette=palette_image,
    )

    result_pixels = np.asarray(quantized_image.convert("RGB"), dtype=np.uint8).copy()
    result_pixels[final_alpha == 0] = 0
    result_image = Image.fromarray(result_pixels, mode="RGB").convert("RGBA")
    result_image.putalpha(Image.fromarray(final_alpha, mode="L"))
    return result_image


class ProcessingCancelled(Exception):
    """Indica o cancelamento cooperativo do processamento Atkinson."""


def apply_atkinson(
    image: Image.Image,
    color_count: int,
    preserve_transparency: bool,
    hard_alpha: bool,
    should_cancel: Callable[[], bool] | None = None,
) -> Image.Image:
    """Aplica dithering Atkinson colorido com uma paleta adaptativa."""
    if should_cancel is not None and should_cancel():
        raise ProcessingCancelled()
    color_count = max(2, min(color_count, 256))
    rgba_image = image.convert("RGBA")
    source_pixels = np.asarray(rgba_image, dtype=np.uint8)
    final_alpha = calculate_final_alpha(
        source_pixels[:, :, 3], preserve_transparency, hard_alpha
    )
    _, palette_colors = create_adaptive_palette(
        source_pixels, final_alpha, color_count
    )
    if palette_colors.size == 0:
        return Image.new("RGBA", rgba_image.size, (0, 0, 0, 0))

    # A LUT de 5 bits evita recalcular a distância para a paleta por pixel.
    color_lut = create_color_lut(palette_colors)

    height, width = final_alpha.shape
    visible_mask = final_alpha > 0
    working_rgb = source_pixels[:, :, :3].astype(np.float32)
    result_rgb = np.zeros_like(source_pixels[:, :, :3])
    neighbors = ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))

    for y_position in range(height):
        if should_cancel is not None and should_cancel():
            raise ProcessingCancelled()
        for x_position in range(width):
            if not visible_mask[y_position, x_position]:
                continue

            current_color = np.clip(working_rgb[y_position, x_position], 0, 255)
            red_index = min(31, max(0, int(current_color[0] * 31 / 255)))
            green_index = min(31, max(0, int(current_color[1] * 31 / 255)))
            blue_index = min(31, max(0, int(current_color[2] * 31 / 255)))
            new_color = color_lut[red_index, green_index, blue_index]
            result_rgb[y_position, x_position] = new_color
            distributed_error = (current_color - new_color.astype(np.float32)) / 8

            for offset_x, offset_y in neighbors:
                neighbor_x = x_position + offset_x
                neighbor_y = y_position + offset_y
                if (
                    0 <= neighbor_x < width
                    and 0 <= neighbor_y < height
                    and visible_mask[neighbor_y, neighbor_x]
                ):
                    working_rgb[neighbor_y, neighbor_x] += distributed_error

    result_image = Image.fromarray(result_rgb, mode="RGB").convert("RGBA")
    result_image.putalpha(Image.fromarray(final_alpha, mode="L"))
    return result_image


def generate_bayer_matrix(size: int) -> np.ndarray:
    """Gera e normaliza uma matriz Bayer de tamanho 2, 4 ou 8."""
    if size not in (2, 4, 8):
        raise ValueError("O tamanho da matriz Bayer deve ser 2, 4 ou 8.")

    matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    while matrix.shape[0] < size:
        matrix = np.block(
            [[4 * matrix, 4 * matrix + 2], [4 * matrix + 3, 4 * matrix + 1]]
        )
    return (matrix + 0.5) / (size * size) - 0.5


def apply_bayer_dithering(
    image: Image.Image,
    color_count: int,
    matrix_size: int,
    preserve_transparency: bool,
    hard_alpha: bool,
    should_cancel: Callable[[], bool] | None = None,
) -> Image.Image:
    """Aplica dithering Bayer colorido sem percorrer pixel por pixel."""
    if should_cancel is not None and should_cancel():
        raise ProcessingCancelled()

    color_count = max(2, min(color_count, 256))
    rgba_image = image.convert("RGBA")
    source_pixels = np.asarray(rgba_image, dtype=np.uint8)
    final_alpha = calculate_final_alpha(
        source_pixels[:, :, 3], preserve_transparency, hard_alpha
    )
    _, palette_colors = create_adaptive_palette(
        source_pixels, final_alpha, color_count
    )
    if should_cancel is not None and should_cancel():
        raise ProcessingCancelled()
    if palette_colors.size == 0:
        return Image.new("RGBA", rgba_image.size, (0, 0, 0, 0))

    color_lut = create_color_lut(palette_colors)
    if should_cancel is not None and should_cancel():
        raise ProcessingCancelled()

    height, width = final_alpha.shape
    normalized_matrix = generate_bayer_matrix(matrix_size)
    threshold_map = normalized_matrix[
        np.arange(height)[:, np.newaxis] % matrix_size,
        np.arange(width)[np.newaxis, :] % matrix_size,
    ]
    levels_per_channel = max(2.0, color_count ** (1.0 / 3.0))
    amplitude = 255.0 / (levels_per_channel - 1.0)
    adjusted_rgb = np.clip(
        source_pixels[:, :, :3].astype(np.float32) + threshold_map[:, :, np.newaxis] * amplitude,
        0,
        255,
    )
    if should_cancel is not None and should_cancel():
        raise ProcessingCancelled()

    lookup_indices = np.clip((adjusted_rgb * 31 / 255).astype(np.intp), 0, 31)
    result_rgb = np.zeros_like(source_pixels[:, :, :3])
    visible_mask = final_alpha > 0
    result_rgb[visible_mask] = color_lut[
        lookup_indices[:, :, 0][visible_mask],
        lookup_indices[:, :, 1][visible_mask],
        lookup_indices[:, :, 2][visible_mask],
    ]
    if should_cancel is not None and should_cancel():
        raise ProcessingCancelled()

    result_image = Image.fromarray(result_rgb, mode="RGB").convert("RGBA")
    result_image.putalpha(Image.fromarray(final_alpha, mode="L"))
    return result_image


def apply_noise_dithering(
    image: Image.Image,
    color_count: int,
    noise_intensity: int,
    preserve_transparency: bool,
    hard_alpha: bool,
    seed: int = 12345,
    cancel_callback: Callable[[], bool] | None = None,
) -> Image.Image:
    """Aplica ruído determinístico antes de mapear cores para a paleta."""
    if cancel_callback is not None and cancel_callback():
        raise ProcessingCancelled()

    color_count = max(2, min(color_count, 256))
    noise_intensity = max(0, min(100, int(noise_intensity)))
    rgba_image = image.convert("RGBA")
    source_pixels = np.asarray(rgba_image, dtype=np.uint8)
    final_alpha = calculate_final_alpha(
        source_pixels[:, :, 3], preserve_transparency, hard_alpha
    )
    if cancel_callback is not None and cancel_callback():
        raise ProcessingCancelled()

    _, palette_colors = create_adaptive_palette(
        source_pixels, final_alpha, color_count
    )
    if cancel_callback is not None and cancel_callback():
        raise ProcessingCancelled()
    if palette_colors.size == 0:
        return Image.new("RGBA", rgba_image.size, (0, 0, 0, 0))

    color_lut = create_color_lut(palette_colors)
    if cancel_callback is not None and cancel_callback():
        raise ProcessingCancelled()

    levels_per_channel = max(2.0, color_count ** (1.0 / 3.0))
    base_amplitude = 255.0 / (levels_per_channel - 1.0)
    amplitude = base_amplitude * (noise_intensity / 100.0)
    rgb_pixels = source_pixels[:, :, :3].astype(np.float32)
    if noise_intensity == 0:
        # Zero significa quantização estável, sem gerar ruído desnecessário.
        adjusted_rgb = rgb_pixels.copy()
    else:
        random_generator = np.random.default_rng(seed)
        noise = random_generator.uniform(
            -1.0, 1.0, size=rgb_pixels.shape
        ).astype(np.float32)
        adjusted_rgb = rgb_pixels + noise * amplitude
    adjusted_rgb = np.nan_to_num(
        adjusted_rgb, nan=0.0, posinf=255.0, neginf=0.0
    )
    adjusted_rgb = np.clip(adjusted_rgb, 0.0, 255.0)
    if cancel_callback is not None and cancel_callback():
        raise ProcessingCancelled()

    lookup_indices = np.clip((adjusted_rgb * 31 / 255).astype(np.intp), 0, 31)
    result_rgb = np.zeros_like(source_pixels[:, :, :3])
    visible_mask = final_alpha > 0
    result_rgb[visible_mask] = color_lut[
        lookup_indices[:, :, 0][visible_mask],
        lookup_indices[:, :, 1][visible_mask],
        lookup_indices[:, :, 2][visible_mask],
    ]
    if cancel_callback is not None and cancel_callback():
        raise ProcessingCancelled()

    result_image = Image.fromarray(result_rgb, mode="RGB").convert("RGBA")
    result_image.putalpha(Image.fromarray(final_alpha, mode="L"))
    return result_image


def create_atkinson_preview(image: Image.Image) -> Image.Image:
    """Cria uma cópia reduzida para a pré-visualização do Atkinson."""
    maximum_dimension = 1200
    largest_dimension = max(image.size)
    if largest_dimension <= maximum_dimension:
        return image.copy()

    scale = maximum_dimension / largest_dimension
    preview_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    # A exportação futura processará a imagem original em segundo plano.
    return image.resize(preview_size, Image.Resampling.LANCZOS)


class ImageProcessingWorker(QObject):
    """Executa Atkinson ou Bayer fora da thread da interface."""

    completed = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)
    saving = Signal(int)

    def __init__(
        self,
        image: Image.Image,
        settings: ProcessingSettings,
        generation: int,
        task_type: str = "preview",
        export_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._image = image
        self._settings = settings
        self._generation = generation
        self._task_type = task_type
        self._export_path = export_path
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Solicita o cancelamento sem acessar widgets."""
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        """Processa e envia somente dados prontos para a janela."""
        try:
            result_image = process_image_with_settings(
                self._image, self._settings, self._cancel_event.is_set
            )
            if self._cancel_event.is_set():
                raise ProcessingCancelled()
            if self._task_type == "export":
                if self._export_path is None:
                    raise ValueError("Caminho de exportação não informado.")
                self.saving.emit(self._generation)
                self._save_png(result_image)
                self.completed.emit(self._generation, str(self._export_path))
                return
        except ProcessingCancelled:
            self.cancelled.emit(self._generation)
        except Exception:
            self.failed.emit(self._generation, traceback.format_exc())
        else:
            self.completed.emit(self._generation, result_image)

    def _save_png(self, image: Image.Image) -> None:
        """Salva primeiro um PNG temporário e depois substitui o destino."""
        if self._export_path is None:
            raise ValueError("Caminho de exportação não informado.")
        temporary_path = self._export_path.with_name(
            f"{self._export_path.stem}.tmp.png"
        )
        try:
            image.convert("RGBA").save(
                temporary_path, format="PNG", optimize=False, compress_level=6
            )
            if self._cancel_event.is_set():
                raise ProcessingCancelled()
            os.replace(temporary_path, self._export_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


class ImagePreviewWidget(QWidget):
    """Mostra uma imagem sobre um fundo quadriculado de transparência."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self.setMinimumSize(200, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: Image.Image | None) -> None:
        """Recebe uma imagem Pillow RGBA e atualiza a pré-visualização."""
        self._image = self._pillow_to_qimage(image) if image is not None else None
        self.update()

    @staticmethod
    def _pillow_to_qimage(image: Image.Image) -> QImage:
        """Converte RGBA do Pillow para QImage sem perder o canal alfa."""
        rgba_data = image.tobytes("raw", "RGBA")
        qimage = QImage(
            rgba_data,
            image.width,
            image.height,
            image.width * 4,
            QImage.Format.Format_RGBA8888,
        )
        # copy() torna o QImage independente do buffer temporário em Python.
        return qimage.copy()

    def paintEvent(self, event: object) -> None:
        """Desenha o quadriculado, a imagem centralizada ou o aviso vazio."""
        del event
        painter = QPainter(self)
        try:
            if self.size().isEmpty():
                return

            self._draw_checkerboard(painter)

            if self._image is None:
                painter.setPen(QColor("#9da1a6"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Nenhuma imagem carregada")
                return

            image_size = self._image.size()
            preview_size = image_size.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            if preview_size.isEmpty():
                return

            target_rect = QRect(0, 0, preview_size.width(), preview_size.height())
            target_rect.moveCenter(self.rect().center())

            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(target_rect, self._image)
        finally:
            painter.end()

    def _draw_checkerboard(self, painter: QPainter) -> None:
        """Desenha quadrados alternados para indicar áreas transparentes."""
        square_size = 16
        light_color = QColor("#c8c8c8")
        dark_color = QColor("#9d9d9d")

        for y_position in range(0, self.height(), square_size):
            for x_position in range(0, self.width(), square_size):
                is_light_square = (x_position // square_size + y_position // square_size) % 2 == 0
                painter.fillRect(
                    x_position,
                    y_position,
                    square_size,
                    square_size,
                    light_color if is_light_square else dark_color,
                )


class ImagePanel(QFrame):
    """Área com título que contém uma pré-visualização de imagem."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imagePanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        layout.addWidget(title_label)

        self.preview = ImagePreviewWidget()
        layout.addWidget(self.preview, 1)


class MainWindow(QMainWindow):
    """Janela principal da aplicação."""

    def __init__(self) -> None:
        super().__init__()
        self.original_image: Image.Image | None = None
        self.result_image: Image.Image | None = None
        self.original_file_path: str | None = None
        self.processing_generation = 0
        self.atkinson_thread: QThread | None = None
        self.atkinson_worker: ImageProcessingWorker | None = None
        self._pending_atkinson_request: tuple[
            int, str, Image.Image, int, int, bool, bool
        ] | None = None
        self._atkinson_processing = False
        self._active_task_type = "preview"
        self._pending_export_request: tuple[int, Image.Image, ProcessingSettings, Path] | None = None
        self._exporting = False
        self.export_generation = 0
        self._active_export_path: Path | None = None
        self._is_closing = False
        self.noise_debounce_timer = QTimer(self)
        self.noise_debounce_timer.setSingleShot(True)
        self.noise_debounce_timer.setInterval(180)

        self.setWindowTitle("Dither App")
        self.resize(1200, 760)
        self.setMinimumSize(900, 600)

        self._create_interface()
        self._apply_styles()
        self._connect_controls()
        self._update_control_states()

    def _create_interface(self) -> None:
        """Monta os painéis e controles da janela."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)

        self.import_button = QPushButton("Importar imagem")
        self.export_button = QPushButton("Exportar PNG")
        self.reset_button = QPushButton("Redefinir")
        self.export_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_image)
        self.export_button.clicked.connect(self.export_png)
        self.reset_button.clicked.connect(self.reset_application)

        toolbar_layout.addWidget(self.import_button)
        toolbar_layout.addWidget(self.export_button)
        toolbar_layout.addWidget(self.reset_button)
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        images_layout = QHBoxLayout()
        images_layout.setSpacing(16)
        self.original_panel = ImagePanel("Imagem original")
        self.result_panel = ImagePanel("Resultado")
        images_layout.addWidget(self.original_panel, 1)
        images_layout.addWidget(self.result_panel, 1)
        content_layout.addLayout(images_layout, 1)

        content_layout.addWidget(self._create_settings_panel())
        main_layout.addLayout(content_layout, 1)

    def _create_settings_panel(self) -> QFrame:
        """Cria o painel lateral de configurações visuais."""
        panel = QFrame()
        panel.setObjectName("settingsPanel")
        panel.setFixedWidth(270)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title_label = QLabel("Configurações")
        title_label.setObjectName("settingsTitle")
        layout.addWidget(title_label)

        layout.addWidget(QLabel("Método de dithering"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            [
                "Nenhum", "Threshold", "Floyd–Steinberg", "Atkinson",
                "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Ruído",
            ]
        )
        layout.addWidget(self.method_combo)

        layout.addWidget(QLabel("Quantidade de cores"))
        self.colors_spinbox = QSpinBox()
        self.colors_spinbox.setRange(2, 256)
        self.colors_spinbox.setValue(16)
        layout.addWidget(self.colors_spinbox)

        self.threshold_label = QLabel("Limiar: 128")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_slider)

        self.noise_label = QLabel("Intensidade do ruído: 50%")
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(50)
        layout.addWidget(self.noise_label)
        layout.addWidget(self.noise_slider)

        self.preserve_alpha_checkbox = QCheckBox("Preservar transparência")
        self.preserve_alpha_checkbox.setChecked(True)
        self.hard_alpha_checkbox = QCheckBox("Alfa rígido para pixel art")
        layout.addWidget(self.preserve_alpha_checkbox)
        layout.addWidget(self.hard_alpha_checkbox)
        layout.addStretch()
        return panel

    def _connect_controls(self) -> None:
        """Conecta os controles que atualizam o resultado."""
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.noise_slider.valueChanged.connect(self._on_noise_intensity_changed)
        self.noise_debounce_timer.timeout.connect(self._apply_debounced_noise_update)
        self.colors_spinbox.valueChanged.connect(self._on_color_count_changed)
        self.preserve_alpha_checkbox.toggled.connect(self.update_result)
        self.hard_alpha_checkbox.toggled.connect(self.update_result)

    def _on_method_changed(self, method: str) -> None:
        """Atualiza o estado dos controles ao trocar de método."""
        if method != "Ruído":
            self.noise_debounce_timer.stop()
        self._update_control_states()
        if method not in (
            "Nenhum", "Threshold", "Floyd–Steinberg", "Atkinson",
            "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Ruído",
        ):
            self.statusBar().showMessage(
                f'"{method}" será implementado posteriormente.', 4000
            )
        else:
            self.statusBar().clearMessage()
        self.update_result()

    def _on_threshold_changed(self, threshold: int) -> None:
        """Mostra o valor atual do limiar e recalcula o resultado."""
        self.threshold_label.setText(f"Limiar: {threshold}")
        self.update_result()

    def _on_noise_intensity_changed(self, intensity: int) -> None:
        """Mostra a porcentagem de ruído e atualiza o resultado quando necessário."""
        self.noise_label.setText(f"Intensidade do ruído: {intensity}%")
        if self.method_combo.currentText() == "Ruído":
            # Invalida o resultado antigo sem criar uma thread a cada passo do slider.
            self.processing_generation += 1
            self._cancel_atkinson_processing()
            self.noise_debounce_timer.start()

    def _apply_debounced_noise_update(self) -> None:
        """Processa Ruído quando o usuário para de mover o slider."""
        if self.method_combo.currentText() == "Ruído":
            self.update_result()

    def _on_color_count_changed(self, color_count: int) -> None:
        """Atualiza os métodos que usam quantidade de cores."""
        del color_count
        if self.method_combo.currentText() in (
            "Floyd–Steinberg", "Atkinson", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Ruído",
        ):
            self.update_result()

    def _update_control_states(self) -> None:
        """Ativa apenas os controles disponíveis para o método atual."""
        method = self.method_combo.currentText()
        is_threshold = method == "Threshold"
        is_noise = method == "Ruído"
        uses_color_count = method in (
            "Floyd–Steinberg", "Atkinson", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Ruído",
        )
        self.threshold_label.setEnabled(is_threshold)
        self.threshold_slider.setEnabled(is_threshold)
        self.colors_spinbox.setEnabled(uses_color_count)
        self.noise_label.setEnabled(is_noise)
        self.noise_slider.setEnabled(is_noise)

    @staticmethod
    def _is_background_method(method: str) -> bool:
        """Indica os métodos atendidos pelo worker compartilhado."""
        return method in ("Atkinson", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Ruído")

    def import_image(self) -> None:
        """Abre uma imagem e a mostra nos dois painéis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar imagem",
            "",
            "Imagens (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not file_path:
            return

        try:
            with Image.open(file_path) as opened_image:
                corrected_image = ImageOps.exif_transpose(opened_image)
                self.original_image = corrected_image.convert("RGBA")
        except (OSError, UnidentifiedImageError) as error:
            QMessageBox.critical(
                self,
                "Não foi possível abrir a imagem",
                f"O arquivo não pôde ser aberto:\n{file_path}\n\nDetalhes: {error}",
            )
            return

        self.original_panel.preview.set_image(self.original_image)
        self.original_file_path = file_path
        self.update_result()
        self.reset_button.setEnabled(self.atkinson_thread is None)
        self.export_button.setEnabled(True)

    def update_result(self) -> None:
        """Recalcula o resultado sempre a partir da imagem original."""
        self.processing_generation += 1
        generation = self.processing_generation
        if self.original_image is None:
            return

        if self.method_combo.currentText() == "Threshold":
            self._cancel_atkinson_processing()
            self.result_image = apply_threshold(
                self.original_image,
                self.threshold_slider.value(),
                self.preserve_alpha_checkbox.isChecked(),
                self.hard_alpha_checkbox.isChecked(),
            )
        elif self.method_combo.currentText() == "Floyd–Steinberg":
            self._cancel_atkinson_processing()
            self.statusBar().showMessage("Processando Floyd–Steinberg...")
            QApplication.processEvents()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.result_image = apply_floyd_steinberg(
                    self.original_image,
                    self.colors_spinbox.value(),
                    self.preserve_alpha_checkbox.isChecked(),
                    self.hard_alpha_checkbox.isChecked(),
                )
                self.statusBar().showMessage(
                    f"Floyd–Steinberg aplicado com {self.colors_spinbox.value()} cores"
                )
            except Exception as error:
                print(f"Erro ao aplicar Floyd–Steinberg: {error}", file=sys.stderr)
                self.result_image = self.original_image.copy()
                self.statusBar().showMessage("Não foi possível aplicar Floyd–Steinberg.")
            finally:
                QApplication.restoreOverrideCursor()
        elif self.method_combo.currentText() in (
            "Atkinson", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Ruído",
        ):
            self._request_atkinson_processing(generation)
            return
        else:
            self._cancel_atkinson_processing()
            self.result_image = self.original_image.copy()

        self.result_panel.preview.set_image(self.result_image)

    def _request_atkinson_processing(self, generation: int) -> None:
        """Inicia ou agenda a versão mais recente do processamento em segundo plano."""
        if self.original_image is None:
            return

        request = (
            generation,
            self.method_combo.currentText(),
            create_atkinson_preview(self.original_image),
            self.colors_spinbox.value(),
            self.noise_slider.value(),
            self.preserve_alpha_checkbox.isChecked(),
            self.hard_alpha_checkbox.isChecked(),
        )
        if self.atkinson_thread is not None:
            self._pending_atkinson_request = request
            self._cancel_atkinson_processing(keep_pending=True)
            return

        self._start_atkinson_worker(request)

    def _start_atkinson_worker(
        self,
        request: tuple[int, str, Image.Image, int, int, bool, bool],
    ) -> None:
        """Cria um worker e mantém suas referências até o fim da thread."""
        (
            generation,
            method,
            preview_image,
            color_count,
            noise_intensity,
            preserve_alpha,
            hard_alpha,
        ) = request
        processing_message = (
            "Processando dithering por ruído..."
            if method == "Ruído"
            else f"Processando {method}..."
        )
        self.statusBar().showMessage(processing_message)
        if not self._atkinson_processing:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._atkinson_processing = True
        QApplication.processEvents()
        self._set_atkinson_processing_state(True)

        settings = ProcessingSettings(
            method=method,
            color_count=color_count,
            threshold=self.threshold_slider.value(),
            noise_intensity=noise_intensity,
            preserve_transparency=preserve_alpha,
            hard_alpha=hard_alpha,
        )

        thread = QThread(self)
        worker = ImageProcessingWorker(
            preview_image,
            settings,
            generation,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_atkinson_completed)
        worker.failed.connect(self._on_atkinson_failed)
        worker.cancelled.connect(self._on_atkinson_cancelled)
        worker.saving.connect(self._on_export_saving)
        worker.completed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.cancelled.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_atkinson_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self.atkinson_thread = thread
        self.atkinson_worker = worker
        thread.start()

    def _cancel_atkinson_processing(self, keep_pending: bool = False) -> None:
        """Solicita cancelamento cooperativo ao worker ativo."""
        if not keep_pending:
            self._pending_atkinson_request = None
        if self.atkinson_worker is not None:
            self.atkinson_worker.request_cancel()

    def _current_settings(self) -> ProcessingSettings:
        """Captura as configurações atuais para um processamento isolado."""
        return ProcessingSettings(
            method=self.method_combo.currentText(),
            color_count=self.colors_spinbox.value(),
            threshold=self.threshold_slider.value(),
            noise_intensity=self.noise_slider.value(),
            preserve_transparency=self.preserve_alpha_checkbox.isChecked(),
            hard_alpha=self.hard_alpha_checkbox.isChecked(),
        )

    def export_png(self) -> None:
        """Pede o destino e agenda a exportação em resolução completa."""
        if self.original_image is None or self._exporting:
            return

        default_name = "imagem_dithered.png"
        if self.original_file_path is not None:
            default_name = f"{Path(self.original_file_path).stem}_dithered.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar PNG", default_name, "PNG (*.png)"
        )
        if not file_path:
            return

        export_path = Path(file_path)
        if export_path.suffix.lower() != ".png":
            export_path = export_path.with_suffix(".png")

        self.export_generation += 1
        self._pending_export_request = (
            self.export_generation,
            self.original_image.copy(),
            self._current_settings(),
            export_path,
        )
        self._set_export_state(True)
        self._pending_atkinson_request = None
        self._cancel_atkinson_processing()
        if self.atkinson_thread is None:
            request = self._pending_export_request
            self._pending_export_request = None
            self._start_export_worker(request)

    def _start_export_worker(
        self,
        request: tuple[int, Image.Image, ProcessingSettings, Path] | None,
    ) -> None:
        """Inicia um worker de exportação depois do fim seguro da prévia."""
        if request is None:
            return
        generation, image, settings, export_path = request
        self._active_task_type = "export"
        self._active_export_path = export_path
        self.statusBar().showMessage(
            "Processando imagem em resolução completa para exportação..."
        )
        if not self._atkinson_processing:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._atkinson_processing = True
        QApplication.processEvents()

        thread = QThread(self)
        worker = ImageProcessingWorker(image, settings, generation, "export", export_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_atkinson_completed)
        worker.failed.connect(self._on_atkinson_failed)
        worker.cancelled.connect(self._on_atkinson_cancelled)
        worker.saving.connect(self._on_export_saving)
        worker.completed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.cancelled.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_atkinson_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.atkinson_thread = thread
        self.atkinson_worker = worker
        thread.start()

    def _on_atkinson_completed(self, generation: int, result_image: Image.Image) -> None:
        """Aplica somente o resultado correspondente à configuração atual."""
        if self._active_task_type == "export":
            if generation == self.export_generation:
                exported_name = Path(str(result_image)).name
                self.statusBar().showMessage(
                    f"PNG exportado com sucesso: {exported_name}", 5000
                )
            return
        if (
            generation != self.processing_generation
            or not self._is_background_method(self.method_combo.currentText())
        ):
            return

        self.result_image = result_image
        self.result_panel.preview.set_image(self.result_image)
        self.statusBar().showMessage(
            (
                f"Ruído aplicado com {self.colors_spinbox.value()} cores e "
                f"intensidade de {self.noise_slider.value()}%"
                if self.method_combo.currentText() == "Ruído"
                else f"{self.method_combo.currentText()} aplicado com "
                f"{self.colors_spinbox.value()} cores"
            )
        )

    def _on_export_saving(self, generation: int) -> None:
        """Atualiza o status quando o worker inicia a gravação do PNG."""
        if self._active_task_type == "export" and generation == self.export_generation:
            self.statusBar().showMessage("Salvando PNG...")

    def _on_atkinson_failed(self, generation: int, error_details: str) -> None:
        """Restaura uma cópia segura quando o worker falha."""
        print(f"Erro no processamento de imagem:\n{error_details}", file=sys.stderr)
        if self._active_task_type == "export":
            if generation == self.export_generation:
                self.statusBar().showMessage("Não foi possível exportar o PNG.")
                QMessageBox.critical(
                    self,
                    "Erro ao exportar PNG",
                    "Não foi possível salvar o PNG solicitado:\n"
                    f"{self._active_export_path}\n\n"
                    "Verifique o caminho, permissões e espaço em disco.",
                )
            return
        if generation != self.processing_generation or self.original_image is None:
            return

        self.result_image = self.original_image.copy()
        self.result_panel.preview.set_image(self.result_image)
        self.statusBar().showMessage("Não foi possível aplicar o método selecionado.")

    def _on_atkinson_cancelled(self, generation: int) -> None:
        """Informa um cancelamento apenas quando ele ainda é relevante."""
        if self._active_task_type == "export" and generation == self.export_generation:
            self.statusBar().showMessage("Exportação cancelada.")
            return
        if generation == self.processing_generation and not self._is_closing:
            self.statusBar().showMessage("Processamento cancelado.")

    def _on_atkinson_thread_finished(self) -> None:
        """Libera referências e inicia, se houver, a última solicitação pendente."""
        finished_thread = self.sender()
        if finished_thread is not self.atkinson_thread:
            return

        self.atkinson_thread = None
        self.atkinson_worker = None
        if self._pending_export_request is not None and not self._is_closing:
            export_request = self._pending_export_request
            self._pending_export_request = None
            self._start_export_worker(export_request)
            return
        pending_request = self._pending_atkinson_request
        self._pending_atkinson_request = None

        if (
            pending_request is not None
            and pending_request[0] == self.processing_generation
            and self._is_background_method(self.method_combo.currentText())
            and not self._is_closing
        ):
            self._start_atkinson_worker(pending_request)
            return

        self._set_atkinson_processing_state(False)
        if self._active_task_type == "export":
            self._set_export_state(False)
            self._active_task_type = "preview"
            self._active_export_path = None
        if self._atkinson_processing:
            QApplication.restoreOverrideCursor()
            self._atkinson_processing = False

    def _set_atkinson_processing_state(self, is_processing: bool) -> None:
        """Mantém apenas os botões de arquivo indisponíveis durante o trabalho."""
        self.import_button.setEnabled(not is_processing)
        self.reset_button.setEnabled(not is_processing and self.original_image is not None)

    def _set_export_state(self, is_exporting: bool) -> None:
        """Bloqueia controles enquanto a exportação usa as opções capturadas."""
        self._exporting = is_exporting
        controls = (
            self.import_button, self.export_button, self.reset_button,
            self.method_combo, self.colors_spinbox, self.threshold_slider,
            self.noise_slider, self.preserve_alpha_checkbox, self.hard_alpha_checkbox,
        )
        if is_exporting:
            for control in controls:
                control.setEnabled(False)
            return

        self.import_button.setEnabled(True)
        self.export_button.setEnabled(self.original_image is not None)
        self.reset_button.setEnabled(self.original_image is not None)
        self.method_combo.setEnabled(True)
        self.preserve_alpha_checkbox.setEnabled(True)
        self.hard_alpha_checkbox.setEnabled(True)
        self._update_control_states()

    def reset_application(self) -> None:
        """Limpa as imagens e restaura os controles aos valores iniciais."""
        self.processing_generation += 1
        self._cancel_atkinson_processing()
        self.noise_debounce_timer.stop()
        self.original_image = None
        self.result_image = None
        self.original_file_path = None
        self.original_panel.preview.set_image(None)
        self.result_panel.preview.set_image(None)
        self.reset_button.setEnabled(False)
        self.export_button.setEnabled(False)

        controls = (
            self.method_combo,
            self.threshold_slider,
            self.noise_slider,
            self.colors_spinbox,
            self.preserve_alpha_checkbox,
            self.hard_alpha_checkbox,
        )
        blockers = [QSignalBlocker(control) for control in controls]
        self.method_combo.setCurrentText("Nenhum")
        self.threshold_slider.setValue(128)
        self.noise_slider.setValue(50)
        self.noise_label.setText("Intensidade do ruído: 50%")
        self.colors_spinbox.setValue(16)
        self.preserve_alpha_checkbox.setChecked(True)
        self.hard_alpha_checkbox.setChecked(False)
        del blockers
        self._update_control_states()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cancela o worker antes de destruir a janela."""
        self._is_closing = True
        self.noise_debounce_timer.stop()
        self._cancel_atkinson_processing()
        if self.atkinson_thread is not None and self.atkinson_thread.isRunning():
            self.atkinson_thread.wait()
        if self._atkinson_processing:
            QApplication.restoreOverrideCursor()
            self._atkinson_processing = False
        event.accept()

    def _apply_styles(self) -> None:
        """Aplica uma aparência escura simples à aplicação."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1e1f22;
                color: #e6e6e6;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3a6ea5;
                border: none;
                border-radius: 5px;
                padding: 8px 14px;
                color: white;
            }
            QPushButton:hover:!disabled { background-color: #4b82ba; }
            QPushButton:disabled { background-color: #3a3d41; color: #858585; }
            #imagePanel, #settingsPanel {
                background-color: #292b2f;
                border: 1px solid #3b3e44;
                border-radius: 8px;
            }
            #panelTitle, #settingsTitle {
                color: #ffffff;
                font-size: 16px;
                font-weight: 600;
            }
            QComboBox, QSpinBox {
                background-color: #202124;
                border: 1px solid #50535a;
                border-radius: 4px;
                padding: 6px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QCheckBox { spacing: 8px; padding: 3px 0; }
            """
        )


def main() -> int:
    """Inicia a aplicação Qt."""
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
