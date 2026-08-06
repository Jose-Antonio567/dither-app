"""Interface inicial do Dither App."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ImagePanel(QFrame):
    """Área reservada para exibir uma imagem."""

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

        # Espaço central onde a imagem será mostrada em etapas futuras.
        placeholder = QLabel("Nenhuma imagem carregada")
        placeholder.setObjectName("placeholder")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder, 1)


class MainWindow(QMainWindow):
    """Janela principal da aplicação."""

    def __init__(self) -> None:
        super().__init__()
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

        import_button = QPushButton("Importar imagem")
        export_button = QPushButton("Exportar PNG")
        reset_button = QPushButton("Redefinir")
        export_button.setEnabled(False)
        reset_button.setEnabled(False)

        toolbar_layout.addWidget(import_button)
        toolbar_layout.addWidget(export_button)
        toolbar_layout.addWidget(reset_button)
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        images_layout = QHBoxLayout()
        images_layout.setSpacing(16)
        images_layout.addWidget(ImagePanel("Imagem original"), 1)
        images_layout.addWidget(ImagePanel("Resultado"), 1)
        content_layout.addLayout(images_layout, 1)

        settings_panel = self._create_settings_panel()
        content_layout.addWidget(settings_panel)
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

        method_label = QLabel("Método de dithering")
        method_combo = QComboBox()
        method_combo.addItems(
            [
                "Nenhum",
                "Threshold",
                "Floyd–Steinberg",
                "Atkinson",
                "Bayer 2x2",
                "Bayer 4x4",
                "Bayer 8x8",
                "Ruído",
            ]
        )
        layout.addWidget(method_label)
        layout.addWidget(method_combo)

        colors_label = QLabel("Quantidade de cores")
        colors_spinbox = QSpinBox()
        colors_spinbox.setRange(2, 256)
        colors_spinbox.setValue(16)
        layout.addWidget(colors_label)
        layout.addWidget(colors_spinbox)

        preserve_alpha_checkbox = QCheckBox("Preservar transparência")
        preserve_alpha_checkbox.setChecked(True)
        hard_alpha_checkbox = QCheckBox("Alfa rígido para pixel art")
        layout.addWidget(preserve_alpha_checkbox)
        layout.addWidget(hard_alpha_checkbox)
        layout.addStretch()

        return panel

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
            #placeholder { color: #9da1a6; font-size: 15px; }
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
