"""Configurações de processamento de imagem."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingSettings:
    """Valores usados por prévias e exportações."""

    method: str
    color_count: int
    threshold: int
    noise_intensity: int
    preserve_transparency: bool
    hard_alpha: bool
    noise_seed: int = 12345
    remove_background: bool = False
    background_color: tuple[int, int, int] = (255, 255, 255)
    background_tolerance: int = 30
    background_connected_only: bool = True
    pixel_scale: int = 1


def get_default_processing_settings() -> ProcessingSettings:
    """Retorna a predefinição interna Normal."""
    return ProcessingSettings("Nenhum", 16, 128, 50, True, False)


__all__ = ["ProcessingSettings", "get_default_processing_settings"]
