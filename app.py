"""Interface inicial do Dither App com importação e pré-visualização."""

import sys

from PIL import Image, ImageOps, UnidentifiedImageError
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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

        self.setWindowTitle("Dither App")
        self.resize(1200, 760)
        self.setMinimumSize(900, 600)

        self._create_interface()
        self._apply_styles()

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

        self.preserve_alpha_checkbox = QCheckBox("Preservar transparência")
        self.preserve_alpha_checkbox.setChecked(True)
        self.hard_alpha_checkbox = QCheckBox("Alfa rígido para pixel art")
        layout.addWidget(self.preserve_alpha_checkbox)
        layout.addWidget(self.hard_alpha_checkbox)
        layout.addStretch()
        return panel

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
            self.result_image = self.original_image.copy()
        except (OSError, UnidentifiedImageError) as error:
            QMessageBox.critical(
                self,
                "Não foi possível abrir a imagem",
                f"O arquivo não pôde ser aberto:\n{file_path}\n\nDetalhes: {error}",
            )
            return

        self.original_panel.preview.set_image(self.original_image)
        self.result_panel.preview.set_image(self.result_image)
        self.reset_button.setEnabled(True)
        self.export_button.setEnabled(False)

    def reset_application(self) -> None:
        """Limpa as imagens e restaura os controles aos valores iniciais."""
        self.original_image = None
        self.result_image = None
        self.original_panel.preview.set_image(None)
        self.result_panel.preview.set_image(None)
        self.reset_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.method_combo.setCurrentText("Nenhum")
        self.colors_spinbox.setValue(16)
        self.preserve_alpha_checkbox.setChecked(True)
        self.hard_alpha_checkbox.setChecked(False)

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
