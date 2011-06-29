;Including header files
!include logiclib.nsh
!include x64.nsh
!include WordFunc.nsh
!include MUI2.nsh
!include UninstallLog.nsh

;Defining compile time constants and necessary variables
!define APP_NAME "Sonic AWE"
!define PUBLISHER "Reep"
!define NVID_VERSION ""
!define INST_FILES ""
!define FILE_NAME ""
!define REG_ROOT HKCU
!define REG_APP_PATH "Software\Reep\Sonic AWE"

;--------------------------------
; Configure UnInstall log to only remove what is installed
;-------------------------------- 
;Set the name of the uninstall log
!define UninstLog "uninstall.log"
Var UninstLog

;Uninstall log file missing.
LangString UninstLogMissing ${LANG_ENGLISH} "${UninstLog} not found!$\r$\nUninstallation cannot proceed!"

;AddItem macro
!define AddItem "!insertmacro AddItem"

;File macro
!define File "!insertmacro File"

;CreateShortcut macro
!define CreateShortcut "!insertmacro CreateShortcut"

;Copy files macro
!define CopyFiles "!insertmacro CopyFiles"

;Rename macro
!define Rename "!insertmacro Rename"

;CreateDirectory macro
!define CreateDirectory "!insertmacro CreateDirectory"

;SetOutPath macro
!define SetOutPath "!insertmacro SetOutPath"

;WriteUninstaller macro
!define WriteUninstaller "!insertmacro WriteUninstaller"

;WriteRegStr macro
!define WriteRegStr "!insertmacro WriteRegStr"

;WriteRegDWORD macro
!define WriteRegDWORD "!insertmacro WriteRegDWORD"

;DxDiag macro
!define Dxdiag "!insertmacro DxDiag"

Section -openlogfile
    CreateDirectory "$INSTDIR"
    IfFileExists "$INSTDIR\${UninstLog}" +3
      FileOpen $UninstLog "$INSTDIR\${UninstLog}" w
    Goto +4
      SetFileAttributes "$INSTDIR\${UninstLog}" NORMAL
      FileOpen $UninstLog "$INSTDIR\${UninstLog}" a
      FileSeek $UninstLog 0 END
SectionEnd
	
;var NVID_VERSION 
Var INSTALLATION_DONE
Var USR_DRIVER_VERSION
Var done
Var StartMenuFolder
  
;Name of the installer
Name "Sonic AWE"

;The file to write
OutFile ${FILE_NAME}

;Require Admin execution level 
RequestExecutionLevel admin

;defining page look&feel
;!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_PAGE_HEADER_TEXT "Sonic AWE Setup"
!define MUI_WELCOMEPAGE_TITLE "Welcome to the Sonic AWE Setup"
!define MUI_TEXT_WELCOME_INFO_TEXT "Welcome to the installation wizard for Sonic AWE. This will install Sonic AWE on your computer. Click Next to proceed"
!define MUI_WELCOMEFINISHPAGE_BITMAP "Side_Banner.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP_NOSTRETCH
!define MUI_ICON "awe256.ico"

; The default installation directory
InstallDir "$PROGRAMFILES\Reep\Sonic AWE"

;Get installation folder from registry if available
InstallDirRegKey ${REG_ROOT} "Software\Reep\Sonic AWE" ""

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
	${SetOutPath} $INSTDIR
		
	;Write the installation path into the registry
	${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "Install Directory" "$INSTDIR"
	${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "AppData" "$LOCALAPPDATA\Reep\Sonic AWE"
	;Write the Uninstall information into the registry
	${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "UninstallString" "$INSTDIR\uninstall.exe"
  
	;Create uninstaller
	${WriteUninstaller} "$INSTDIR\Uninstall.exe"
	
	Strcpy $INSTALLATION_DONE "0"

	;Retrieving user's driver version
	${CreateDirectory} "$LOCALAPPDATA\Reep\Sonic AWE"
	DetailPrint "Launching dxdiag for NVIDIA driver compatibility check"
	${DxDiag}
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
		${if} $8 == "30" 
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
			${File} ${INST_FILES} "*.*"
			Strcpy $INSTALLATION_DONE "1"
			Goto done
		${elseif} $R0 == 2 
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
  ${CreateShortCut} "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\Sonicawe.exe" "" "" ""

SectionEnd

Section "Start Menu Shortcut"
	;create desktop shortcut
	!insertmacro MUI_STARTMENU_WRITE_BEGIN Application
		${CreateDirectory} "$SMPROGRAMS\$StartMenuFolder"
		${CreateShortCut} "$SMPROGRAMS\$StartMenuFolder\Sonic AWE.lnk" "$INSTDIR\Sonicawe.exe" "" "" ""
		${CreateShortCut} "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "$INSTDIR\Uninstall.exe" "" "" ""
	!insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section "Uninstall"
	;check windows version
	${If} ${RunningX64}
		SetRegView 64
	${EndIf}
	
	;Can't uninstall if the log is missing!
	IfFileExists "$INSTDIR\${UninstLog}" +3
    MessageBox MB_OK|MB_ICONSTOP "$(UninstLogMissing)"
      Abort

	Push $R0
	Push $R1
	Push $R2
	SetFileAttributes "$INSTDIR\${UninstLog}" NORMAL
	FileOpen $UninstLog "$INSTDIR\${UninstLog}" r
	StrCpy $R1 -1
	
	GetLineCount:
		ClearErrors
		FileRead $UninstLog $R0
		IntOp $R1 $R1 + 1
		StrCpy $R0 $R0 -2
		Push $R0   
		IfErrors 0 GetLineCount
	 
	Pop $R0
	 
	LoopRead:
		StrCmp $R1 0 LoopDone
		Pop $R0
	 
		IfFileExists "$R0\*.*" 0 +3
		  RMDir $R0  #is dir
		Goto +9
		IfFileExists $R0 0 +3
		  Delete $R0 #is file
		Goto +6
		StrCmp $R0 "${REG_ROOT} ${REG_APP_PATH}" 0 +3
		  DeleteRegKey ${REG_ROOT} "${REG_APP_PATH}" #is Reg Element
		Goto +3
		StrCmp $R0 "${REG_ROOT} ${REG_APP_PATH}" 0 +2
		  DeleteRegKey ${REG_ROOT} "${REG_APP_PATH}" #is Reg Element
	 
		IntOp $R1 $R1 - 1
		Goto LoopRead
	LoopDone:
		FileClose $UninstLog
		Delete "$INSTDIR\${UninstLog}"
		Pop $R2
		Pop $R1
		Pop $R0  
		  
	Delete "$INSTDIR\Uninstall.exe"
	
	StrCpy $0 "$INSTDIR"
	Call un.DeleteDirIfEmpty

	StrCpy $0 "$SMPROGRAMS\Reep"
	Call un.DeleteDirIfEmpty
	
	StrCpy $0 "$LOCALAPPDATA\Reep"
	Call un.DeleteDirIfEmpty
	
	DeleteRegKey HKCU "Software\Reep\Sonic AWE"
	DeleteRegKey /ifempty HKCU "Software\Reep"

SectionEnd

Function un.DeleteDirIfEmpty
  FindFirst $R0 $R1 "$0\*.*"
  strcmp $R1 "." 0 NoDelete
   FindNext $R0 $R1
   strcmp $R1 ".." 0 NoDelete
    ClearErrors
    FindNext $R0 $R1
    IfErrors 0 NoDelete
     FindClose $R0
     Sleep 1000
     RMDir "$0"
  NoDelete:
   FindClose $R0
FunctionEnd

