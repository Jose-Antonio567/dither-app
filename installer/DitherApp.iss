#define MyAppName "Dither App"
#define MyAppVersion "0.9.0"
#define MyAppVersionLabel "0.9.0 Beta"
#define MyAppPublisher "Tiny Glitch"
#define MyAppExeName "DitherApp.exe"

[Setup]
AppId={{64850F7E-7186-46EC-9E33-4D6D91D7EB26}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersionLabel}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Dither App
UsePreviousAppDir=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
SetupIconFile=..\assets\app_icon.ico
UninstallDisplayIcon={app}\DitherApp.exe
Compression=lzma2
SolidCompression=yes
OutputDir=..\release\installer
OutputBaseFilename=DitherApp-Setup-0.9.0-beta-win64
VersionInfoVersion=0.9.0.0
VersionInfoCompany=Tiny Glitch
VersionInfoProductName=Dither App
VersionInfoDescription=Instalador do Dither App
VersionInfoCopyright=Copyright 2026 Tiny Glitch
CloseApplications=yes

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\dist\DitherApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Dither App"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\Dither App"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Criar um atalho na Área de Trabalho"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Executar Dither App"; Flags: nowait postinstall skipifsilent
