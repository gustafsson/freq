#!/bin/bash
set -e

packagefullname="${packagename}_${versiontag}_win32"
filename="${packagefullname}_setup.exe"
nsistemplate="sonic/sonicawe/dist/package-win/Sonicawe_template.nsi"
nsisscript="sonic/sonicawe/dist/package-win/Sonicawe.nsi"
nsiswriter="sonic/sonicawe/dist/package-win/Nsi_Writer.exe"
licensefile="license.txt"

# make vcbuild called by msbuild detect changes in headers
PATH="/c/Program Files (x86)/Microsoft Visual Studio 9.0/Common7/IDE:${PATH}"

msbuildparams="//property:Configuration=Release //verbosity:detailed sonic.sln"

cd ../..
echo "========================== Building ==========================="
echo "Building ${packagename} ${versiontag}"

echo qmaketarget: $qmaketarget
(cd gpumisc && qmake $qmaketarget)
(cd sonicawe && qmake $qmaketarget)
qmake $qmaketarget

if [ "Y" == "${rebuildall}" ]; then
  "C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean $msbuildparams
fi

rm -f gpumisc/release/gpumisc.lib
touch sonicawe/sawe/configuration/configuration.cpp

"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" $msbuildparams
cp sonicawe/release/sonicawe.exe sonicawe/release/sonicawe-cpu.exe


echo "========================== Building ==========================="
echo "Building ${packagename} cuda ${versiontag}"
qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"

echo qmaketarget: $qmaketarget
(cd gpumisc && qmake $qmaketarget)
(cd sonicawe && qmake $qmaketarget)
qmake $qmaketarget

if [ "Y" == "${rebuildall}" ]; then
  "C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean $msbuildparams
fi

rm -f gpumisc/release/gpumisc.lib
touch sonicawe/sawe/configuration/configuration.cpp

"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" $msbuildparams
cp sonicawe/release/sonicawe.exe sonicawe/release/sonicawe-cuda.exe


echo "========================== Building ==========================="
echo "Building Sonic AWE ${packagename} Launcher"

cd sonicawe/dist/package-win/launcher
qmake "DEFINES+=PACKAGE=\"${packagename}\""
"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean //p:Configuration=Release launcher.sln
"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //p:Configuration=Release launcher.sln
cd ../../../..


echo "========================== Installer =========================="
echo "Creating Windows installer file: $(pwd)/$filename for package $packagefullname"
cd ..
rm -rf $filename
rm -rf $packagefullname
cp -r winlib/sonicawe_snapshot_win32_base $packagefullname
cp sonic/sonicawe/release/sonicawe-cpu.exe "$packagefullname/${packagename}-cpu.exe"
cp sonic/sonicawe/release/sonicawe-cuda.exe "$packagefullname/${packagename}-cuda.exe"
cp sonic/sonicawe/dist/package-win/launcher/release/launcher.exe "$packagefullname/${packagename}.exe"
cp -r sonic/sonicawe/matlab $packagefullname/matlab
cp sonic/sonicawe/license/$licensefile $packagefullname
cp sonic/sonicawe/dist/package-win/awe_256.ico $packagefullname


#echo " - Executing dxdiag for Nvidia driver version minimum requirement"
#CMD //C dxdiag //x %CD%\\dxdiag.xml
#if [ -f dxdiag.xml ]; then
#nvid_version=`sed -e '/DriverVersion/ !d' -e 's!<DriverVersion>\([^<]*\)</DriverVersion>!\~&\~!' dxdiag.xml | awk -F"~" '{print $2}' | cut -f2 -d">" | cut -f1 -d"<"`
#else
#echo Nvidia driver version could not be read because dxdiag xml file was not found. WARNING, version value is set to \"1.0.0.0\" any version of Nvidia drivers will be recognized as compatible.
#nvid_version="1.0.0.0"
#fi


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

# append \ to paths for ${File}
sed -i.backup -r "s/(^\\$\{File\}.*) (.*$)/\1\\\\ \2/g" $nsisscript

#sed="sed -i.backup -e"
sed="sed -i.backup"
#sed="sed -i"" --regexp-extended"

prettyname=
for word in $(echo ${packagename} | sed "s/-/ /g"); do
	prettyname="${prettyname} $(echo $word | awk -F: '{ print toupper(substr ($1,1,1)) substr ($1,2) }')"
done;

prettyname=$(echo ${prettyname} | sed "s/Sonicawe/Sonic AWE/g");
echo $prettyname

if [ -z "${target}" ]; then 
$sed "s/\!define APP\_NAME \".*\"/\!define APP\_NAME \"${prettyname}\"/" $nsisscript
$sed "s/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \".*\"/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \"Side\_Banner\.bmp\"/" $nsisscript
$sed "s/\!define EXE\_NAME \".*\"/\!define EXE\_NAME \"${packagename}.exe\"/" $nsisscript
else
$sed "s/\!define APP\_NAME \".*\"/\!define APP\_NAME \"${prettyname}\"/" $nsisscript
$sed "s/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \".*\"/\!define MUI\_WELCOMEFINISHPAGE\_BITMAP \"${target}\-Side\_Banner\.bmp\"/" $nsisscript
$sed "s/\!define EXE\_NAME \".*\"/\!define EXE\_NAME \"${packagename}.exe\"/" $nsisscript
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


