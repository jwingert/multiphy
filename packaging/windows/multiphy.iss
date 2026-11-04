#ifndef AppVersion
  #error AppVersion must be provided by build.py.
#endif
#ifndef SourceDir
  #error SourceDir must be provided by build.py.
#endif
#ifndef OutputDir
  #error OutputDir must be provided by build.py.
#endif
#ifndef IconFile
  #error IconFile must be provided by build.py.
#endif

[Setup]
AppId={{21E02744-F6D8-4F08-9EC7-9527F00F6C24}
AppName=multiphy
AppVersion={#AppVersion}
AppPublisher=multiphy contributors
AppPublisherURL=https://github.com/jwingert/multiphy
AppSupportURL=https://github.com/jwingert/multiphy/issues
AppUpdatesURL=https://github.com/jwingert/multiphy/releases
DefaultDirName={localappdata}\Programs\multiphy
DefaultGroupName=multiphy
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=multiphy-{#AppVersion}-windows-x64-setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\multiphy-gui.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
LicenseFile={#SourceDir}\LICENSE.txt

[Tasks]
Name: desktopicon; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Remove obsolete bundled libraries on upgrade. This is application-owned code;
; recording/project directories are never created inside the installation.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\multiphy"; Filename: "{app}\multiphy-gui.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\multiphy"; Filename: "{app}\multiphy-gui.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\multiphy-gui.exe"; Description: "Open multiphy"; Flags: nowait postinstall skipifsilent

; No data-directory cleanup: projects, recordings and preferences belong to the
; user and survive application upgrades and uninstall.
