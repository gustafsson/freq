#!/bin/bash
set -e

packagefullname="${packagename}_${versiontag}_win32"
filename="${packagename}_setup.exe"
nsistemplate="sonic/sonicawe/dist/package-win/Sonicawe_template.nsi"
nsisscript="sonic/sonicawe/dist/package-win/Sonicawe.nsi"
nsiswriter="sonic/sonicawe/dist/package-win/Nsi_Writer.exe"
licensefile="license.txt"

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${packagename}"  
echo qmaketarget: $qmaketarget

if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
	"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean //p:Configuration=Release sonic.sln
	cd gpumisc
	qmake $qmaketarget
	cd ../sonicawe
	qmake $qmaketarget
	cd ..
	qmake $qmaketarget
	"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean //p:Configuration=Release sonic.sln
else
  rm -f sonicawe/release/sonicawe.exe
fi
"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //m:2 //p:Configuration=Release sonic.sln
	
echo "========================== Installer =========================="
echo "Creating Windows installer file: $(pwd)/$filename for package $packagefullname"
cd ..
rm -rf $filename
rm -rf $packagefullname
cp -r sonicawe_snapshot_win32_base $packagefullname
cp sonic/sonicawe/dist/package-win/sonicawe.exe.manifest $packagefullname
if [ -z "${target}" ]; then 
	cp sonic/sonicawe/release/sonicawe.exe $packagefullname
else
	cp sonic/sonicawe/release/sonicawe.exe $packagefullname/$packagefullname".exe"
fi
cp -r sonic/sonicawe/matlab $packagefullname/matlab
cp sonic/sonicawe/license/$licensefile $packagefullname
cp sonic/sonicawe/dist/package-win/awe_256.ico $packagefullname

#Executing dxdiag for Nvidia driver version minimum requirement
CMD //C dxdiag //x %CD%\\dxdiag.xml
if [ -f dxdiag.xml ]; then
nvid_version=`sed -e '/DriverVersion/ !d' -e 's!<DriverVersion>\([^<]*\)</DriverVersion>!\~&\~!' dxdiag.xml | awk -F"~" '{print $2}' | cut -f2 -d">" | cut -f1 -d"<"`
else
echo Nvidia driver version could not be read because dxdiag xml file was not found. WARNING, version value is set to \"1.0.0.0\" any version of Nvidia drivers will be recognized as compatible.
nvid_version="1.0.0.0"
fi

echo " - execute NsiWriter.exe to create and fill the Sonicawe.nsi script"
nsistemplate=`pwd`\/$nsistemplate
nsistemplate=`echo $nsistemplate | sed 's@\\/c\\/@C:\\\\\\\@'`
nsistemplate=`echo $nsistemplate | sed 's@\\/@\\\\\\\@g'`
nsisscriptwin=`pwd`\/$nsisscript
nsisscriptwin=`echo $nsisscriptwin | sed 's@\\/c\\/@C:\\\\\\\@'`
nsisscriptwin=`echo $nsisscriptwin | sed 's@\\/@\\\\\\\@g'`
instfilepath=`pwd`\/$packagefullname
instfilepathwin=`echo $instfilepath | sed 's@\\/c\\/@C:\\\\@'`
instfilepathwin=`echo $instfilepathwin | sed 's@\\/@\\\\@g'`
instfilepath=`echo $instfilepath | sed 's@\\/c\\/@C:\\\\\\\@'`
instfilepath=`echo $instfilepath | sed 's@\\/@\\\\\\\@g'`
$nsiswriter "$nsistemplate" "$nsisscriptwin" "$instfilepathwin"

#sed="sed -i.backup -e"
sed="sed -i.backup"
#sed="sed -i"" --regexp-extended"

if [ -z "${target}" ]; then 
$sed "s/\!define APP\_NAME \".*\"/\!define APP\_NAME \"Sonic AWE\"/" $nsisscript 
$sed "s/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \".*\"/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \"Side\_Banner\.bmp\"/" $nsisscript 
$sed "s/\!define EXE\_NAME \".*\"/\!define EXE\_NAME \"sonicawe.exe\"/" $nsisscript 
elif [ "${target}" == "reader" ]; then
$sed "s/\!define APP\_NAME \".*\"/\!define APP\_NAME \"Sonic AWE Reader\"/" $nsisscript 
$sed "s/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \".*\"/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \"reader\-Side\_Banner\.bmp\"/" $nsisscript 
$sed "s/\!define EXE\_NAME \".*\"/\!define EXE\_NAME \"sonicawe_reader.exe\"/" $nsisscript 
else
$sed "s/\!define APP\_NAME \".*\"/\!define APP\_NAME \"Sonic AWE ${target}\"/" $nsisscript 
$sed "s/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \".*\"/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \"${target}\-Side\_Banner\.bmp\"/" $nsisscript
$sed "s/\!define EXE\_NAME \".*\"/\!define EXE\_NAME \"sonicawe_${target}.exe\"/" $nsisscript 
fi

echo " - inserting filename, version and nvidia version number in NSIS script"
$sed "s/\!define SA\_VERSION \".*\"/\!define SA\_VERSION \"${versiontag}\"/" $nsisscript 
$sed "s/\!define NVID\_VERSION \".*\"/\!define NVID\_VERSION \"$nvid_version\"/" $nsisscript 
$sed "s/\!define INST\_FILES \".*\"/\!define INST\_FILES \"$instfilepath\"/" $nsisscript 
$sed "s/\!define FILE\_NAME \".*\"/\!define FILE\_NAME \"$filename\"/" $nsisscript 
licensepath=`pwd`\/$packagefullname\/license.txt
licensepath=`echo $licensepath | sed 's@\\/c\\/@C:\\\\\\\@'`
licensepath=`echo $licensepath | sed 's@\\/@\\\\\\\@g'`
$sed "s/\!insertmacro MUI\_PAGE\_LICENSE \".*\"/\!insertmacro MUI\_PAGE\_LICENSE \"$licensepath\"/" $nsisscript 

echo " - compiling installer"
#Check for NSIS
type -P makensis >& /dev/null || { echo "NSIS is not installed.  Aborting installer compilation. Please install Nsis to avoid this error" >&2; exit 1; }

#compile & move installer
makensis $nsisscript
mv sonic/sonicawe/dist/package-win/$filename sonic/sonicawe/dist/$filename

#clean sonicawe.nsi for git consistency
rm $nsisscript

echo "installer compiled"

cd sonic/sonicawe/dist


