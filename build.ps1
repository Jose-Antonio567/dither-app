$ErrorActionPreference = 'Stop'
if (-not (Test-Path app.py) -or -not (Test-Path DitherApp.spec)) { throw 'Execute este script na raiz do projeto.' }
python --version
python -m PyInstaller --version
python -m unittest discover -s tests -v
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
python -m PyInstaller --noconfirm --clean DitherApp.spec
if (-not (Test-Path dist\DitherApp\DitherApp.exe)) { throw 'Executável não foi criado.' }
Write-Host "Build criada em: $((Resolve-Path dist\DitherApp\DitherApp.exe).Path)"
