"""Ponto de entrada do Dither App."""

import sys

from PySide6.QtWidgets import QApplication

from dither_app.main_window import MainWindow


def main() -> int:
    """Inicia a aplicação Qt."""
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
