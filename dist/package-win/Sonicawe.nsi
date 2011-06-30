;Including header files
!include logiclib.nsh
!include x64.nsh
!include WordFunc.nsh
!include MUI2.nsh
!include UninstallLog.nsh

;Defining compile time constants and necessary variables
!define APP_NAME "Sonic AWE"
!define PUBLISHER "REEP"
!define SA_VERSION ""
!define NVID_VERSION ""
!define INST_FILES ""
!define FILE_NAME ""
!define REG_ROOT HKCU
!define REG_APP_PATH "Software\REEP\Sonic AWE"

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
!define MUI_ICON "awe256.ico"

; The default installation directory
InstallDir "$PROGRAMFILES\REEP\Sonic AWE"

;Get installation folder from registry if available
InstallDirRegKey ${REG_ROOT} "Software\REEP\Sonic AWE" ""

;Show installation details
ShowInstDetails show

; Pages to display during the installation process
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE ""
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY

;Start Menu Folder Page Configuration
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKCU" 
!define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\REEP\Sonic AWE" 
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
	
	Strcpy $INSTALLATION_DONE "0"

	;Retrieving user's driver version
	${CreateDirectory} "$LOCALAPPDATA\REEP\Sonic AWE"
	DetailPrint "Launching dxdiag for NVIDIA driver compatibility check"
	${DxDiag}
	exec 'dxdiag /x $LOCALAPPDATA\REEP\Sonic AWE\dxdiag.xml'
	Strcpy $done "0"
	Strcpy $8 "0"
	${While} $done == "0"
		Sleep 5000	
		IntOp $8 $8 + 1
		IfFileExists "$LOCALAPPDATA\REEP\Sonic AWE\dxdiag.xml" 0 +7
			nsisXML::Create
			nsisXML::Load "$LOCALAPPDATA\REEP\Sonic AWE\dxdiag.xml"
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
		Strcpy $INSTALLATION_DONE "0"
		Goto done
	${elseif} $USR_DRIVER_VERSION != ""
		${VersionCompare} $USR_DRIVER_VERSION ${NVID_VERSION} $R0
		${if} $R0 <= 1  	
		
			;Write the installation path into the registry
			${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "Install Directory" "$INSTDIR"
			${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "AppData" "$LOCALAPPDATA\REEP\Sonic AWE"
			
			;Write the Uninstall information into the registry
			${WriteRegStr} "${REG_ROOT}" "${REG_APP_PATH}" "UninstallString" "$INSTDIR\uninstall.exe"
	
			;Write the Uninstall information to the registry for add/remove program 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "DisplayName" "Sonic AWE -- Visualization based signal analysis"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "DisplayIcon" "$\"$INSTDIR\awe_256.ico$\""
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "InstallLocation" "$\"$INSTDIR$\""
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "DisplayVersion" "${SA_VERSION}"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "Publisher" "REEP"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "HelpLink" "www.sonicawe.com"
						 
			${WriteRegDWORD} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "NoModify" "1"
						 
			${WriteRegDWORD} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "NoRepair" "1"
						 
			${WriteRegStr} "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE" \
						 "URLInfoAbout" "www.sonicawe.com"
		  
			;Create uninstaller
			${WriteUninstaller} "$INSTDIR\Uninstall.exe"
			
			${File} ${INST_FILES}\matlab\examples amplify.m
			${File} ${INST_FILES}\matlab\examples amplify2.m
			${File} ${INST_FILES}\matlab\examples convolve.m
			${File} ${INST_FILES}\matlab\examples fantracker.m
			${File} ${INST_FILES}\matlab\examples lowpass.m
			${File} ${INST_FILES}\matlab\examples markclicks.m
			${File} ${INST_FILES}\matlab\examples plotalarm.m
			${File} ${INST_FILES}\matlab\examples plotamplitude.m
			${File} ${INST_FILES}\matlab\examples plotwaveform.m
			${File} ${INST_FILES}\matlab\examples verifyregulation.m
			${File} ${INST_FILES}\matlab a440.m
			${File} ${INST_FILES}\matlab encavi.sh
			${File} ${INST_FILES}\matlab heightmap.frag.m
			${File} ${INST_FILES}\matlab matlabfilter.m
			${File} ${INST_FILES}\matlab plotspectra2d.m
			${File} ${INST_FILES}\matlab plot_producevideo.m
			${File} ${INST_FILES}\matlab read_csv_from_sonicawe.m
			${File} ${INST_FILES}\matlab sawe_compute_cwt.m
			${File} ${INST_FILES}\matlab sawe_datestr.m
			${File} ${INST_FILES}\matlab sawe_discard.m
			${File} ${INST_FILES}\matlab sawe_extract_cwt.m
			${File} ${INST_FILES}\matlab sawe_extract_cwt_time.m
			${File} ${INST_FILES}\matlab sawe_filewatcher.asv
			${File} ${INST_FILES}\matlab sawe_filewatcher.m
			${File} ${INST_FILES}\matlab sawe_getdatainfo.m
			${File} ${INST_FILES}\matlab sawe_loadbuffer.m
			${File} ${INST_FILES}\matlab sawe_loadchunk.m
			${File} ${INST_FILES}\matlab sawe_plot.m
			${File} ${INST_FILES}\matlab sawe_plot2.m
			${File} ${INST_FILES}\matlab sawe_savebuffer.m
			${File} ${INST_FILES}\matlab sawe_savechunk.m
			${File} ${INST_FILES}\matlab soundtoavi.sh
			${File} ${INST_FILES} credits.txt
			${File} ${INST_FILES} cudart32_32_16.dll
			${File} ${INST_FILES} cufft32_32_16.dll
			${File} ${INST_FILES} glew-license.txt
			${File} ${INST_FILES} glew32.dll
			${File} ${INST_FILES} glut32.dll
			${File} ${INST_FILES} hdf5-license.txt
			${File} ${INST_FILES} hdf5dll.dll
			${File} ${INST_FILES} hdf5_hldll.dll
			${File} ${INST_FILES} libsndfile-1.dll
			${File} ${INST_FILES} license.txt
			${File} ${INST_FILES} Microsoft.VC80.CRT.manifest
			${File} ${INST_FILES} Microsoft.VC90.CRT.manifest
			${File} ${INST_FILES} msvcm80.dll
			${File} ${INST_FILES} msvcm90.dll
			${File} ${INST_FILES} msvcp80.dll
			${File} ${INST_FILES} msvcp90.dll
			${File} ${INST_FILES} msvcr80.dll
			${File} ${INST_FILES} msvcr90.dll
			${File} ${INST_FILES} portaudio-license.txt
			${File} ${INST_FILES} portaudio_x86.dll
			${File} ${INST_FILES} qt4-license.txt
			${File} ${INST_FILES} QtCore4.dll
			${File} ${INST_FILES} QtGui4.dll
			${File} ${INST_FILES} QtOpenGL4.dll
			${File} ${INST_FILES} selected_tone.m
			${File} ${INST_FILES} selection.wav
			${File} ${INST_FILES} sndfile-license.txt
			${File} ${INST_FILES} sonicawe.exe
			${File} ${INST_FILES} sonicawe.exe.manifest
			${File} ${INST_FILES} SonicAWE_Icon.png
			${File} ${INST_FILES} zlib1.dll
			${File} ${INST_FILES} awe_256.ico

			CreateDirectory $INSTDIR\matlab
			CopyFiles $INSTDIR\a440.m $INSTDIR\matlab\a440.m
			CopyFiles $INSTDIR\encavi.sh $INSTDIR\matlab\encavi.sh
			CopyFiles $INSTDIR\heightmap.frag.m $INSTDIR\matlab\heightmap.frag.m
			CopyFiles $INSTDIR\matlabfilter.m $INSTDIR\matlab\matlabfilter.m
			CopyFiles $INSTDIR\plotspectra2d.m $INSTDIR\matlab\plotspectra2d.m
			CopyFiles $INSTDIR\plot_producevideo.m $INSTDIR\matlab\plot_producevideo.m
			CopyFiles $INSTDIR\read_csv_from_sonicawe.m $INSTDIR\matlab\read_csv_from_sonicawe.m
			CopyFiles $INSTDIR\sawe_compute_cwt.m $INSTDIR\matlab\sawe_compute_cwt.m
			CopyFiles $INSTDIR\sawe_datestr.m $INSTDIR\matlab\sawe_datestr.m
			CopyFiles $INSTDIR\sawe_discard.m $INSTDIR\matlab\sawe_discard.m
			CopyFiles $INSTDIR\sawe_extract_cwt.m $INSTDIR\matlab\sawe_extract_cwt.m
			CopyFiles $INSTDIR\sawe_extract_cwt_time.m $INSTDIR\matlab\sawe_extract_cwt_time.m
			CopyFiles $INSTDIR\sawe_filewatcher.asv $INSTDIR\matlab\sawe_filewatcher.asv
			CopyFiles $INSTDIR\sawe_filewatcher.m $INSTDIR\matlab\sawe_filewatcher.m
			CopyFiles $INSTDIR\sawe_getdatainfo.m $INSTDIR\matlab\sawe_getdatainfo.m
			CopyFiles $INSTDIR\sawe_loadbuffer.m $INSTDIR\matlab\sawe_loadbuffer.m
			CopyFiles $INSTDIR\sawe_loadchunk.m $INSTDIR\matlab\sawe_loadchunk.m
			CopyFiles $INSTDIR\sawe_plot.m $INSTDIR\matlab\sawe_plot.m
			CopyFiles $INSTDIR\sawe_plot2.m $INSTDIR\matlab\sawe_plot2.m
			CopyFiles $INSTDIR\sawe_savebuffer.m $INSTDIR\matlab\sawe_savebuffer.m
			CopyFiles $INSTDIR\sawe_savechunk.m $INSTDIR\matlab\sawe_savechunk.m
			CopyFiles $INSTDIR\soundtoavi.sh $INSTDIR\matlab\soundtoavi.sh
			
			CreateDirectory $INSTDIR\matlab\examples
			CopyFiles $INSTDIR\amplify.m $INSTDIR\matlab\examples\amplify.m
			CopyFiles $INSTDIR\amplify2.m $INSTDIR\matlab\examples\amplify2.m
			CopyFiles $INSTDIR\convolve.m $INSTDIR\matlab\examples\convolve.m
			CopyFiles $INSTDIR\fantracker.m $INSTDIR\matlab\examples\fantracker.m
			CopyFiles $INSTDIR\lowpass.m $INSTDIR\matlab\examples\lowpass.m
			CopyFiles $INSTDIR\markclicks.m $INSTDIR\matlab\examples\markclicks.m
			CopyFiles $INSTDIR\plotalarm.m $INSTDIR\matlab\examples\plotalarm.m
			CopyFiles $INSTDIR\plotamplitude.m $INSTDIR\matlab\examples\plotamplitude.m
			CopyFiles $INSTDIR\plotwaveform.m $INSTDIR\matlab\examples\plotwaveform.m
			CopyFiles $INSTDIR\verifyregulation.m $INSTDIR\matlab\examples\verifyregulation.m
			
			FileWrite $UninstLog "$INSTDIR\matlab\examples\amplify.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\amplify2.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\convolve.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\fantracker.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\lowpass.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\markclicks.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\plotalarm.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\plotamplitude.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\plotwaveform.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples\verifyregulation.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\a440.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\encavi.sh $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\heightmap.frag.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\matlabfilter.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\plotspectra2d.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\plot_producevideo.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\read_csv_from_sonicawe.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_compute_cwt.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_datestr.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_discard.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_extract_cwt.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_extract_cwt_time.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_filewatcher.asv $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_filewatcher.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_getdatainfo.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_loadbuffer.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_loadchunk.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_plot.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_plot2.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_savebuffer.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\sawe_savechunk.m $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\soundtoavi.sh $\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab\examples$\r$\n"
			FileWrite $UninstLog "$INSTDIR\matlab$\r$\n"
			
			Strcpy $INSTALLATION_DONE "1"
			Goto done
		${elseif} $R0 == 2 
			MessageBox MB_OKCANCEL "Your Nvidia driver version $USR_DRIVER_VERSION is too old and you might encounter issues running Sonic AWE. \ 
			$\nChoose cancel to abort the installation or OK to download newer drivers." IDOK downloadDrivers 
			Strcpy $INSTALLATION_DONE "0"
			Goto done
		${else}
			messageBox MB_OK "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE"
			Strcpy $INSTALLATION_DONE "0"
			Goto done
		${endif}
	${else}
		messageBox MB_OK "Nvidia drivers could not be verified. Please make sure you have the latest drivers installed in order to run Sonic AWE"
			Strcpy $INSTALLATION_DONE "0"
			Goto done
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
	${if} $INSTALLATION_DONE == "1"
		;create desktop shortcut
		${CreateShortCut} "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\Sonicawe.exe" "" "" ""
	${endif}
SectionEnd

Section "Start Menu Shortcut"
	${if} $INSTALLATION_DONE == "1"
		;create desktop shortcut
		!insertmacro MUI_STARTMENU_WRITE_BEGIN Application
			${CreateDirectory} "$SMPROGRAMS\$StartMenuFolder"
			${CreateShortCut} "$SMPROGRAMS\$StartMenuFolder\Sonic AWE.lnk" "$INSTDIR\Sonicawe.exe" "" "" ""
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
	
	StrCpy $0 "$INSTDIR\matlab"
	Call un.DeleteDirIfEmpty
	
	StrCpy $0 "$INSTDIR\matlab\examples"
	Call un.DeleteDirIfEmpty
	
	StrCpy $0 "$INSTDIR"
	Call un.DeleteDirIfEmpty

	StrCpy $0 "$SMPROGRAMS\REEP"
	Call un.DeleteDirIfEmpty
	
	StrCpy $0 "$LOCALAPPDATA\REEP"
	Call un.DeleteDirIfEmpty
	
	DeleteRegKey "${REG_ROOT}" "Software\REEP\Sonic AWE"
	DeleteRegKey /ifempty "${REG_ROOT}" "Software\REEP"
	DeleteRegKey "${REG_ROOT}" "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sonic AWE"

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

