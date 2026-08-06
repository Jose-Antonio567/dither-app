"""Funções de processamento de imagem do Dither App."""

from dither_app.main_window import (
    ProcessingCancelled, apply_alpha_settings, apply_atkinson, apply_bayer_dithering,
    apply_floyd_steinberg, apply_noise_dithering, apply_threshold,
    calculate_final_alpha, create_adaptive_palette, create_color_lut,
    generate_bayer_matrix, process_image_with_settings, remove_background_by_color,
)

__all__ = [name for name in globals() if name.startswith("apply_") or name in {
    "ProcessingCancelled", "calculate_final_alpha", "create_adaptive_palette",
    "create_color_lut", "generate_bayer_matrix", "process_image_with_settings",
    "remove_background_by_color",
}]
