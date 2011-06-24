;Including header files
!include logiclib.nsh
!include x64.nsh
!include WordFunc.nsh
!include MUI2.nsh

;Defining compile time constants and necessary variables
!define APP_NAME "Sonic AWE"
!define PUBLISHER "Reep"
!define NVID_VERSION ""
!define INST_FILES ""
!define FILE_NAME ""

;var NVID_VERSION 
Var INSTALLATION_DONE
Var USR_DRIVER_VERSION
Var done
Var StartMenuFolder
  
; Name of the installer
Name "Sonic AWE"

; The file to write
OutFile ${FILE_NAME}

;defining page look&feel
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_PAGE_HEADER_TEXT "Sonic AWE Setup"
!define MUI_WELCOMEPAGE_TITLE "Welcome to the Sonic AWE Setup"
!define MUI_TEXT_WELCOME_INFO_TEXT "Welcome to the installation wizard for Sonic AWE. This will install Sonic AWE on your computer. Click Next to proceed"

; The default installation directory
InstallDir "$PROGRAMFILES\Reep\Sonic AWE"

;Get installation folder from registry if available
InstallDirRegKey HKCU "Software\Reep\Sonic AWE" ""

;Show installation details
ShowInstDetails show

; Pages to display during the installation process
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE ""
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY

;Start Menu Folder Page Configuration
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKCU" 
!define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\Reep\Sonic AWE" 
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "Start Menu Folder"
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "${PUBLISHER}\${APP_NAME}"
  
!insertmacro MUI_PAGE_STARTMENU Application $StartMenuFolder

!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; The stuff to install
Section "Application Files (required)"
	;check windows version
	${If} ${RunningX64}
		SetRegView 64
	${EndIf}
	
	; Set output path to the installation directory.
	SetOutPath $INSTDIR
	
	;Store installation folder
	WriteRegStr HKCU "Software\Reep\Sonic AWE" "" $INSTDIR
	WriteRegStr HKCU "Software\Reep\Sonic AWE" "AppData" "$LOCALAPPDATA\Reep\Sonic AWE"
  
	;Create uninstaller
	WriteUninstaller "$INSTDIR\Uninstall.exe"
	
	Strcpy $INSTALLATION_DONE "0"

	;Retrieving user's driver version
	CreateDirectory "$LOCALAPPDATA\Reep\Sonic AWE"
	DetailPrint "Launching dxdiag for NVIDIA driver compatibility check"
	exec 'dxdiag /x $LOCALAPPDATA\Reep\Sonic AWE\dxdiag.xml'
	Strcpy $done "0"
	Strcpy $8 "0"
	${While} $done == "0"
		Sleep 5000	
		IntOp $8 $8 + 1
		IfFileExists "$LOCALAPPDATA\Reep\Sonic AWE\dxdiag.xml" 0 +7
			nsisXML::Create
			nsisXML::Load "$LOCALAPPDATA\Reep\Sonic AWE\dxdiag.xml"
			nsisXML::select '/DxDiag/DisplayDevices/DisplayDevice/DriverVersion'
			nsisXML::getText
			Strcpy $USR_DRIVER_VERSION "$3" 
			Strcpy $done "1"
		${if} $8 == "15" 
			DetailPrint "DxDiag -- Time Out"
			Strcpy $done "1"
		${EndIf}	
	${Endwhile}
	
	;Comparing driver version
	${if} $USR_DRIVER_VERSION == ""
		messageBox MB_OK "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE"
	${elseif} $USR_DRIVER_VERSION != ""
		${VersionCompare} $USR_DRIVER_VERSION ${NVID_VERSION} $R0
		${if} $R0 <= 1  
			File /r ${INST_FILES}\*.*
			Strcpy $INSTALLATION_DONE "1"
			Goto done
		${elseif} $R0 == 2 
			;messageBox MB_OK "Your driver version $0 is too old and you might encounter issues running Sonic AWE. Please make sure you visit www.nvidia.com and install the latest drivers available."			
			MessageBox MB_OKCANCEL "Your Nvidia driver version $USR_DRIVER_VERSION is too old and you might encounter issues running Sonic AWE. \ 
			$\nChoose cancel to abort the installation or OK to download newer drivers." IDOK downloadDrivers 
			Strcpy $INSTALLATION_DONE "0"
			Goto done
		${else}
			messageBox MB_OK "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE"
		${endif}
	${else}
		messageBox MB_OK "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE"
	${endif}
  

	downloadDrivers: 
		ExecShell "open" "http://www.nvidia.com/Download/index.aspx?lang=en-us" 
		Strcpy $INSTALLATION_DONE "1"
	done: 
		${if} $INSTALLATION_DONE == "0"
			DetailPrint "The installation did not complete"
			Abort	
		${endif}
SectionEnd

Section "Desktop Icon"

  ;create desktop shortcut
  CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\Sonicawe.exe" ""

SectionEnd

Section "Start Menu Item"
	;create desktop shortcut
	!insertmacro MUI_STARTMENU_WRITE_BEGIN Application
		CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
		CreateShortCut "$SMPROGRAMS\$StartMenuFolder\Sonic AWE.lnk" "$INSTDIR\Sonicawe.exe"
		CreateShortCut "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
	!insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section "Uninstall"
	;check windows version
	${If} ${RunningX64}
		SetRegView 64
	${EndIf}

	Delete "$INSTDIR\Uninstall.exe"
	
	RMDir "$INSTDIR"

	!insertmacro MUI_STARTMENU_GETFOLDER Application $StartMenuFolder
	ReadRegStr $0 HKCU "Software\Reep\Sonic AWE" "Start Menu Folder"

	Delete "$SMPROGRAMS\$0\Sonic AWE.lnk"
	Delete "$SMPROGRAMS\$0\Uninstall.lnk"
	Delete "$DESKTOP\${APP_NAME}.lnk" 
	Delete "$LOCALAPPDATA\Reep\Sonic AWE"
	Delete "$LOCALAPPDATA\Reep"
	RMDir "$SMPROGRAMS\$StartMenuFolder"

	DeleteRegKey /ifempty HKCU "Software\Reep\Sonic AWE"
	DeleteRegKey /ifempty HKCU "Software\Reep"

SectionEnd
