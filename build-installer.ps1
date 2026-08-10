$ErrorActionPreference = 'Stop'
if (-not (Test-Path app.py) -or -not (Test-Path build.ps1) -or -not (Test-Path installer\DitherApp.iss) -or -not (Test-Path assets\app_icon.ico)) { throw 'Execute este script na raiz do projeto e confirme os arquivos de release.' }
& .\build.ps1
if (-not (Test-Path dist\DitherApp\DitherApp.exe)) { throw 'A build onedir não foi criada.' }
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (-not $iscc) { foreach ($path in @('C:\Program Files\Inno Setup 7\ISCC.exe','C:\Program Files (x86)\Inno Setup 7\ISCC.exe','C:\Program Files (x86)\Inno Setup 6\ISCC.exe','C:\Program Files\Inno Setup 6\ISCC.exe')) { if (Test-Path $path) { $iscc = $path; break } } }
if (-not $iscc) { throw 'Inno Setup não foi encontrado. Instale o Inno Setup 6 ou 7 para gerar o instalador.' }
python -m compileall app.py dither_app tests tools
python -m unittest discover -s tests -v
$installer = 'release\installer\DitherApp-Setup-0.11.0-beta-win64.exe'
if (Test-Path $installer) { Remove-Item -Force $installer }
& $iscc installer\DitherApp.iss
if (-not (Test-Path $installer)) { throw 'O instalador não foi criado.' }
$hash = Get-FileHash -Algorithm SHA256 $installer
"Arquivo: DitherApp-Setup-0.11.0-beta-win64.exe`nAlgoritmo: SHA256`nHash: $($hash.Hash)" | Set-Content -Encoding UTF8 release\installer\SHA256.txt
Write-Host "Instalador: $((Resolve-Path $installer).Path)"; Write-Host "SHA-256: $($hash.Hash)"
