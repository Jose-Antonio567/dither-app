"""Ponto de entrada do Dither App."""

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from dither_app.main_window import MainWindow
from dither_app.resources import app_icon_path
from dither_app.styles import DARK_STYLESHEET
from dither_app.version import APP_NAME, APP_VERSION, ORGANIZATION_NAME


def main() -> int:
    """Inicia a aplicação Qt."""
    application = QApplication(sys.argv)
    application.setOrganizationName(ORGANIZATION_NAME)
    application.setApplicationName(APP_NAME)
    application.setApplicationDisplayName(APP_NAME)
    application.setApplicationVersion(APP_VERSION)
    application.setStyleSheet(DARK_STYLESHEET)
    icon_path = app_icon_path()
    if icon_path.is_file():
        application.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
