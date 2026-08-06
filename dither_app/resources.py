"""Localização de recursos tanto no Python quanto no PyInstaller."""

import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """Retorna um recurso empacotado sem depender do diretório atual."""
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base_path / relative_path


def app_icon_path() -> Path:
    """Retorna o caminho opcional do ícone do aplicativo."""
    return resource_path("assets/app_icon.ico")
