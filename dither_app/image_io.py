"""Operações de I/O de imagens usadas pelo aplicativo."""

from pathlib import Path

from PIL import Image, ImageOps


def load_image(file_path: str) -> Image.Image:
    """Abre, corrige EXIF e converte uma imagem para RGBA."""
    with Image.open(file_path) as opened_image:
        return ImageOps.exif_transpose(opened_image).convert("RGBA")


def default_export_name(file_path: str | None) -> str:
    """Cria o nome sugerido para exportação PNG."""
    return f"{Path(file_path).stem}_dithered.png" if file_path else "imagem_dithered.png"


def ensure_png_extension(file_path: str) -> Path:
    """Acrescenta a extensão PNG quando ela não foi informada."""
    path = Path(file_path)
    return path if path.suffix.lower() == ".png" else path.with_suffix(".png")
