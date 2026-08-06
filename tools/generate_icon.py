"""Gera o ICO do aplicativo a partir da arte-fonte PNG."""

from pathlib import Path
import sys

from PIL import Image, ImageOps

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_ico(source_path: Path, destination_path: Path) -> None:
    """Converte um PNG em ICO quadrado com margem transparente."""
    with Image.open(source_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image).convert("RGBA")
    canvas_size = max(image.size)
    canvas = Image.new("RGBA", (canvas_size, canvas_size))
    canvas.alpha_composite(image, ((canvas_size - image.width) // 2, (canvas_size - image.height) // 2))
    margin = round(canvas_size * 0.06)
    content_size = max(1, canvas_size - margin * 2)
    content = canvas.resize((content_size, content_size), Image.Resampling.LANCZOS)
    padded = Image.new("RGBA", (canvas_size, canvas_size))
    padded.alpha_composite(content, ((canvas_size - content_size) // 2, (canvas_size - content_size) // 2))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    padded.save(destination_path, format="ICO", sizes=[(size, size) for size in ICON_SIZES])


def main() -> int:
    source = PROJECT_ROOT / "assets" / "app_icon_source.png"
    destination = PROJECT_ROOT / "assets" / "app_icon.ico"
    if not source.is_file():
        print("Arquivo assets/app_icon_source.png não encontrado.", file=sys.stderr)
        return 1
    generate_ico(source, destination)
    with Image.open(destination) as icon:
        sizes = getattr(icon, "ico", None)
        included = sizes.sizes() if sizes is not None else []
        print(f"Ícone criado: {destination}")
        print(f"Tamanho: {destination.stat().st_size} bytes")
        print(f"Resoluções: {included}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
