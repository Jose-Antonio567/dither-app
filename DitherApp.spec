# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

root = Path(SPECPATH)
icon = root / "assets" / "app_icon.ico"
datas = [(str(icon), "assets")] if icon.is_file() else []

a = Analysis([str(root / "app.py")], pathex=[str(root)], binaries=[], datas=datas,
             hiddenimports=[], hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=["tests"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [], name="DitherApp",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
          console=False, icon=str(icon) if icon.is_file() else None,
          version=str(root / "build_config" / "version_info.txt"))
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name="DitherApp")
