;Including header files
!include logiclib.nsh
!include x64.nsh
!include WordFunc.nsh
!include MUI2.nsh

;Defining compile time constants and necessary variables
!define NVID_VERSION "8.16.11.8933"
!define INST_FILES "sonicawe_0.2011.05.31-snapshot_win32"
!define FILE_NAME "sonicawe_0.2011.05.31-snapshot_win32_setup.exe"

;var NVID_VERSION 
var INSTALLATION_DONE

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

ShowInstDetails show

; Pages to display during the installation process
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "C:\Users\fatcomp\Reep\sonicawe\sonicawe_0.2011.05.31-snapshot_win32\license.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; The stuff to install
Section "Application Files (required)"
	; Set output path to the installation directory.
	SetOutPath $INSTDIR
	
	Strcpy $INSTALLATION_DONE "0"
;	Strcpy $NVID_VERSION "8.17.12.6099"
	
	;check windows version
	${If} ${RunningX64}
		SetRegView 64
	${EndIf}
	
	;Get NVIDIA version from ini file. This version will be compared against the user's drivers
	;${GetSection} "C:\Work\Sonic\SONICAWE_NSIS.ini" "Nvidia build info" "get_nvid_version"	
	
	;Retrieving user's driver version
	readRegStr $0 HKLM "SOFTWARE\NVIDIA Corporation\Installer" version
	
	${if} $0 == ""
		messageBox MB_OK "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE"
	${elseif} $0 != ""
		${VersionCompare} $0 ${NVID_VERSION} $R0
		${if} $R0 <= 1  
			messageBox MB_OK "version $0 is ok"
			File /r ${INST_FILES}\*.*
			Strcpy $INSTALLATION_DONE "1"
			Goto done
		${elseif} $R0 == 2 
			;messageBox MB_OK "Your driver version $0 is too old and you might encounter issues running Sonic AWE. Please make sure you visit www.nvidia.com and install the latest drivers available."			
			MessageBox MB_OKCANCEL "Your Nvidia driver version $0 is too old and you might encounter issues running Sonic AWE. \ 
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
