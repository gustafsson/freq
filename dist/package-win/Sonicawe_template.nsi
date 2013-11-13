;Including header files
!include logiclib.nsh
!include x64.nsh
!include WordFunc.nsh
!include MUI2.nsh
; get UninstallLog.nsh from http://nsis.sourceforge.net/Uninstall_only_installed_files
!addincludedir include
!include UninstallLog.nsh

;Defining compile time constants and necessary variables
!define APP_NAME ""
!define EXE_NAME ""
!define PUBLISHER "MuchDifferent"
!define SA_VERSION ""
!define NVID_VERSION ""
!define INST_FILES ""
!define FILE_NAME ""
!define REG_ROOT HKCU
!define REG_APP_PATH "Software\MuchDifferent\Sonic AWE"

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
Name "${APP_NAME}"

;The file to write
OutFile ${FILE_NAME}

;Require Admin execution level 
RequestExecutionLevel admin

;defining page look&feel
;!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_PAGE_HEADER_TEXT "${APP_NAME} Setup"
!define MUI_WELCOMEPAGE_TITLE "Welcome to the ${APP_NAME} Setup"
!define MUI_TEXT_WELCOME_INFO_TEXT "Welcome to the installation wizard for ${APP_NAME}. This will install ${APP_NAME} on your computer. Click Next to proceed"
!define MUI_WELCOMEFINISHPAGE_BITMAP "Side_Banner.bmp"
!define MUI_ICON "awe32.ico"

; The default installation directory
InstallDir "$PROGRAMFILES\${PUBLISHER}\${APP_NAME}"

;Get installation folder from registry if available
InstallDirRegKey ${REG_ROOT} "Software\${PUBLISHER}\${APP_NAME}" ""

; Pages to display during the installation process
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE ""
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY

;Start Menu Folder Page Configuration
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKCU" 
!define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\${PUBLISHER}\${APP_NAME}" 
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "Start Menu Folder"
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "${PUBLISHER}\${APP_NAME}"

;Adding some branding!  
BrandingText "by MuchDifferent" 
!define MUI_CUSTOMFUNCTION_GUIINIT onGUIInit   
  
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
	
	Strcpy $INSTALLATION_DONE "0"

	;Retrieving user's driver version
;	${CreateDirectory} "$LOCALAPPDATA\${PUBLISHER}\${APP_NAME}"
;	DetailPrint "Launching dxdiag for NVIDIA driver compatibility check"
;	${DxDiag}
;	exec 'dxdiag /x $LOCALAPPDATA\${PUBLISHER}\${APP_NAME}\dxdiag.xml'
;	Strcpy $done "0"
;	Strcpy $8 "0"
;	${While} $done == "0"
;		Sleep 5000	
;		IntOp $8 $8 + 1
;		IfFileExists "$LOCALAPPDATA\${PUBLISHER}\${APP_NAME}\dxdiag.xml" 0 +7
;			nsisXML::Create
;			nsisXML::Load "$LOCALAPPDATA\${PUBLISHER}\${APP_NAME}\dxdiag.xml"
;			nsisXML::select '/DxDiag/DisplayDevices/DisplayDevice/DriverVersion'
;			nsisXML::getText
;			Strcpy $USR_DRIVER_VERSION "$3" 
;			Strcpy $done "1"
;		${if} $8 == "30" 
;			DetailPrint "DxDiag -- Time Out"
;			Strcpy $done "1"
;		${EndIf}	
;	${Endwhile}
	
	Strcpy $USR_DRIVER_VERSION "${NVID_VERSION}"
	
	;Comparing driver version
;	${if} $USR_DRIVER_VERSION == ""
;		messageBox MB_OK|MB_ICONEXCLAMATION "Nvidia drivers could not be verified. Please make sure your hardware meets the requirements to run Sonic AWE and install the latest Nvidia drivers. \
;		                 $\n$\nPlease visit www.MuchDifferent.com for more information \
;						 $\n$\nThe installer will now quit"
;		Strcpy $INSTALLATION_DONE "0"
;		Goto done
;	${elseif} $USR_DRIVER_VERSION != ""
	${if} 1 == 1 
		; TODO this comparsion doesn't work for Quadro GPUs if you're building with a Geforce GPU as the driver versions differ. But they both support CUDA 3.0.
		;${VersionCompare} $USR_DRIVER_VERSION ${NVID_VERSION} $R0

		; don't abort installation, but warn user
;		${if} $R0 == 2
;			MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION "Your Nvidia drivers, version $USR_DRIVER_VERSION, are too old and you might encounter issues running Sonic AWE. \ 
;			$\nChoose OK to download newer drivers now. You will have to restart the installation afterwards." IDOK downloadDrivers 
;		${elseif} $R0 > 1
;			messageBox MB_OK|MB_ICONEXCLAMATION "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE"
;		${endif}

		${if} 1 == 1 
		;$R0 <= 1
		
			;Write the installation path into the registry
			${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "Install Directory" "$INSTDIR"
			${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "AppData" "$LOCALAPPDATA\${PUBLISHER}\${APP_NAME}"
			
			;Write the Uninstall information into the registry
			${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "UninstallString" "$INSTDIR\uninstall.exe"
	
			;Write the Uninstall information to the registry for add/remove program 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "DisplayName" "${APP_NAME} -- Visualization based signal analysis"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "DisplayIcon" "$\"$INSTDIR\awe_256.ico$\""
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "InstallLocation" "$\"$INSTDIR$\""
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "DisplayVersion" "${SA_VERSION}"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "Publisher" "${PUBLISHER}"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "HelpLink" "www.sonicawe.com"
						 
			${WriteRegDWORD} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "NoModify" "1"
						 
			${WriteRegDWORD} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "NoRepair" "1"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
						 "URLInfoAbout" "www.sonicawe.com"
		  
			;Create uninstaller
			${WriteUninstaller} "$INSTDIR\Uninstall.exe"
			
			;Insert files here
			
			Strcpy $INSTALLATION_DONE "1"
			Goto done
		${endif}
	${else}
		messageBox MB_OK|MB_ICONSTOP "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE \
		$\n$\nThe installer will now quit"
			Strcpy $INSTALLATION_DONE "0"
			Goto done
	${endif}
  
	downloadDrivers: 
		ExecShell "open" "http://www.nvidia.com/Download/index.aspx?lang=en-us" 
		Strcpy $INSTALLATION_DONE "1"
	done: 
		${if} $INSTALLATION_DONE == "0"
			DetailPrint "The installation could not complete"
			Quit	
		${endif}
SectionEnd

Section "Desktop Icon"
	${if} $INSTALLATION_DONE == "1"
		;create desktop shortcut
		${CreateShortCut} "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${EXE_NAME}" "" "" ""
	${endif}
SectionEnd

Section "Start Menu Shortcut"
	${if} $INSTALLATION_DONE == "1"
		;create desktop shortcut
		!insertmacro MUI_STARTMENU_WRITE_BEGIN Application
			${CreateDirectory} "$SMPROGRAMS\$StartMenuFolder"
			${CreateShortCut} "$SMPROGRAMS\$StartMenuFolder\${APP_NAME}.lnk" "$INSTDIR\${EXE_NAME}" "" "" ""
			${CreateShortCut} "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "$INSTDIR\Uninstall.exe" "" "" ""
		!insertmacro MUI_STARTMENU_WRITE_END
	${endif}
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
	
	;Insert Uninstall folders command here
	
	StrCpy $0 "$INSTDIR"
	Call un.DeleteDirIfEmpty

	StrCpy $0 "$SMPROGRAMS\${PUBLISHER}"
	Call un.DeleteDirIfEmpty
	
	StrCpy $0 "$LOCALAPPDATA\${PUBLISHER}"
	Call un.DeleteDirIfEmpty
	
	DeleteRegKey "${REG_ROOT}" "Software\${PUBLISHER}\${APP_NAME}"
	DeleteRegKey /ifempty "${REG_ROOT}" "Software\${PUBLISHER}"
	DeleteRegKey "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

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

Function onGUIInit
	;Aero::Apply
	;BrandingURL::Set /NOUNLOAD "0" "0" "200" "http://www.MuchDifferent.com"
FunctionEnd

Function .onGUIEnd
	;BrandingURL::Unload
FunctionEnd
