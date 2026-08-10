$ErrorActionPreference = 'Stop'
if (-not (Test-Path app.py) -or -not (Test-Path DitherApp-OneFile.spec)) { throw 'Execute este script na raiz do projeto.' }
if (-not (Test-Path assets\app_icon_source.png)) { throw 'Arquivo assets/app_icon_source.png não encontrado.' }
python --version
python -m PyInstaller --version
python tools\generate_icon.py
if (-not (Test-Path assets\app_icon.ico)) { throw 'ICO não foi criado.' }
python -m compileall app.py dither_app tests tools
python -m unittest discover -s tests -v
if (Test-Path build\onefile) { Remove-Item -Recurse -Force build\onefile }
if (Test-Path dist\DitherApp.exe) { Remove-Item -Force dist\DitherApp.exe }
python -m PyInstaller --noconfirm --clean DitherApp-OneFile.spec
if (-not (Test-Path dist\DitherApp.exe)) { throw 'Executável onefile não foi criado.' }
$releaseName = 'DitherApp-0.11.0-beta-win64'
$releaseRoot = Join-Path 'release' $releaseName
if (Test-Path $releaseRoot) { Remove-Item -Recurse -Force $releaseRoot }
New-Item -ItemType Directory -Force $releaseRoot | Out-Null
Copy-Item dist\DitherApp.exe $releaseRoot
@"
Dither App 0.11.0 Beta
Desenvolvido por Tiny Glitch

Aplicativo dedicado à aplicação de efeitos de dithering em imagens.
Formatos de entrada: PNG, JPG, JPEG, WebP e BMP.
Formato de saída: PNG com suporte a transparência.

Este aplicativo é portátil e não precisa de instalação.
Para executar: abra DitherApp.exe.
Configurações de interface são armazenadas pelo Windows usando QSettings.
Na primeira abertura, o aplicativo pode demorar alguns segundos para iniciar.
"@ | Set-Content -Encoding UTF8 (Join-Path $releaseRoot 'LEIA-ME.txt')
$hash = Get-FileHash -Algorithm SHA256 (Join-Path $releaseRoot 'DitherApp.exe')
"Arquivo: DitherApp.exe`nAlgoritmo: SHA256`nHash: $($hash.Hash)" | Set-Content -Encoding UTF8 (Join-Path $releaseRoot 'SHA256.txt')
$zip = Join-Path 'release' "$releaseName.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $releaseRoot -DestinationPath $zip
Write-Host "Release criada em: $releaseRoot"
