"""Ponto de entrada do Dither App."""

import sys

from PySide6.QtWidgets import QApplication

from dither_app.main_window import MainWindow
from dither_app.styles import DARK_STYLESHEET


def main() -> int:
    """Inicia a aplicação Qt."""
    application = QApplication(sys.argv)
    application.setOrganizationName("Tiny Glitch")
    application.setApplicationName("Dither App")
    application.setApplicationDisplayName("Dither App")
    application.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
