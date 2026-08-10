"""Persistência das predefinições criadas pelo usuário."""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from dither_app.settings import ProcessingSettings, get_default_processing_settings

SCHEMA_VERSION = 1
NORMAL_PRESET_NAME = "Normal"


@dataclass(frozen=True)
class UserPreset:
    name: str
    settings: ProcessingSettings
    schema_version: int = SCHEMA_VERSION


def preset_storage_path() -> Path:
    """Retorna o local persistente, fora da pasta da aplicação."""
    directory = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )
    return Path(directory) / "presets.json"


def settings_to_dict(settings: ProcessingSettings) -> dict[str, object]:
    return {
        "method": settings.method, "color_count": settings.color_count,
        "threshold": settings.threshold, "noise_intensity": settings.noise_intensity,
        "noise_seed": settings.noise_seed, "pixel_scale": settings.pixel_scale,
        "preserve_transparency": settings.preserve_transparency,
        "hard_alpha": settings.hard_alpha, "remove_background": settings.remove_background,
        "background_color": list(settings.background_color),
        "background_tolerance": settings.background_tolerance,
        "background_connected_only": settings.background_connected_only,
    }


def settings_from_dict(data: object) -> ProcessingSettings:
    """Lê configurações JSON com limites compatíveis com a interface."""
    if not isinstance(data, dict):
        raise ValueError("Configurações inválidas.")
    default = get_default_processing_settings()
    method = data.get("method", default.method)
    color = data.get("background_color", list(default.background_color))
    if not isinstance(method, str) or not isinstance(color, list) or len(color) != 3:
        raise ValueError("Campos de predefinição inválidos.")
    try:
        rgb = tuple(max(0, min(255, int(value))) for value in color)
        return ProcessingSettings(
            method=method, color_count=max(2, min(256, int(data.get("color_count", default.color_count)))),
            threshold=max(0, min(255, int(data.get("threshold", default.threshold)))),
            noise_intensity=max(0, min(100, int(data.get("noise_intensity", default.noise_intensity)))),
            preserve_transparency=bool(data.get("preserve_transparency", default.preserve_transparency)),
            hard_alpha=bool(data.get("hard_alpha", default.hard_alpha)),
            noise_seed=int(data.get("noise_seed", default.noise_seed)),
            remove_background=bool(data.get("remove_background", default.remove_background)),
            background_color=rgb, background_tolerance=max(0, min(255, int(data.get("background_tolerance", default.background_tolerance)))),
            background_connected_only=bool(data.get("background_connected_only", default.background_connected_only)),
            pixel_scale=max(1, min(12, int(data.get("pixel_scale", default.pixel_scale)))),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Valores de predefinição inválidos.") from error


def validate_preset_name(name: str) -> str:
    normalized = name.strip()
    if (not normalized or len(normalized) > 60 or normalized.casefold() == NORMAL_PRESET_NAME.casefold()
            or any(character.isspace() is False and ord(character) < 32 for character in normalized)):
        raise ValueError("Nome de predefinição inválido.")
    return normalized


class PresetManager:
    """Gerencia presets sem acessar widgets."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or preset_storage_path()
        self.incompatible_schema = False
        self.load_error = False

    def load(self) -> list[UserPreset]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
                self.incompatible_schema = True
                self.load_error = True
                print("Arquivo de predefinições possui schema incompatível.", file=sys.stderr)
                return []
            presets: list[UserPreset] = []
            known: set[str] = set()
            for entry in payload.get("presets", []):
                try:
                    name = validate_preset_name(entry.get("name", ""))
                    if name.casefold() in known:
                        continue
                    presets.append(UserPreset(name, settings_from_dict(entry.get("settings"))))
                    known.add(name.casefold())
                except (AttributeError, ValueError):
                    continue
            return sorted(presets, key=lambda preset: preset.name.casefold())
        except (OSError, json.JSONDecodeError) as error:
            self.load_error = True
            print(f"Não foi possível carregar predefinições: {error}", file=sys.stderr)
            return []

    def save(self, presets: list[UserPreset]) -> None:
        names: set[str] = set()
        for preset in presets:
            name = validate_preset_name(preset.name)
            if name.casefold() in names:
                raise ValueError("Nomes de predefinição duplicados.")
            names.add(name.casefold())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name("presets.tmp.json")
        payload = {"schema_version": SCHEMA_VERSION, "presets": [
            {"name": preset.name, "settings": settings_to_dict(preset.settings)} for preset in presets
        ]}
        try:
            temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
