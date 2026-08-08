"""Modelos e utilitários para processamento em lote."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class BatchImageItem:
    file_path: Path
    file_name: str
    width: int
    height: int
    format_name: str
    has_transparency: bool
    thumbnail: Image.Image | None = None


def is_supported_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def make_batch_item(file_path: str | Path) -> BatchImageItem:
    path = Path(file_path).resolve()
    with Image.open(path) as image:
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        thumbnail = rgba.copy()
        thumbnail.thumbnail((64, 64), Image.Resampling.LANCZOS)
        return BatchImageItem(path, path.name, rgba.width, rgba.height,
                              image.format or "Desconhecido", "A" in rgba.getbands(), thumbnail)


def available_output_path(directory: Path, source_path: Path) -> Path:
    stem = f"{source_path.stem}_dithered"
    candidate = directory / f"{stem}.png"
    number = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{number}.png"
        number += 1
    return candidate
