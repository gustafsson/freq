#!/bin/bash
set -e

packagename="${packagename}_${versiontag}_win32"
filename="${packagename}_setup.exe"
packagefullname="tmp/${packagename}"
nsistemplate="dist/package-win/sonicawe_template.nsi"
nsisscript="dist/package-win/sonicawe.nsi"
nsiswriter="dist/package-win/Nsi_Writer.exe"
licensefile="license.txt"

if ls "/c/Program Files (x86)" >& /dev/null; then
	programfiles="/c/Program Files (x86)";
else
	programfiles="/c/Program Files";
fi

# make vcbuild called by msbuild detect changes in headers
if ls "$programfiles/Microsoft Visual Studio 9.0/Common7" >& /dev/null; then
	PATH="$programfiles/Microsoft Visual Studio 9.0/Common7/IDE:${PATH}"
else
	echo "Couldn't find Visual Studio 2008."
	echo "Visual Studio 2008 Express can downloaded at:"
	echo "http://www.microsoft.com/visualstudio/en-us/products/2008-editions/express"
	false
fi

if ls "$programfiles/NSIS" >& /dev/null; then
	PATH="$programfiles/NSIS:${PATH}"
else
	echo "Couldn't find NSIS (Nullsoft Scriptable Install System)."
	echo "NSIS can be downloaded at:"
	echo "http://nsis.sourceforge.net/Download"
	false
fi


msbuildparams="//property:Configuration=Release //verbosity:detailed sonicawe.sln"

pushd ..
echo "========================== Building ==========================="
echo "Building ${packagename} ${versiontag}"

echo qmaketarget: $qmaketarget
(cd lib/gpumisc && qmake $qmaketarget) || false
(cd src && qmake $qmaketarget) || false
qmake $qmaketarget

if [ "Y" == "${rebuildall}" ]; then
  "C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean $msbuildparams
fi

rm -f lib/gpumisc/release/gpumisc.lib
touch src/sawe/configuration/configuration.cpp

"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" $msbuildparams
cp src/release/sonicawe.exe src/release/sonicawe-cpu.exe

echo "========================== Building ==========================="
echo "Building ${packagename} cuda ${versiontag}"
if [ ! -z "$CUDA_BIN_PATH" ]; then
	qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"

	echo qmaketarget: $qmaketarget
	qmake -r $qmaketarget

	if [ "Y" == "${rebuildall}" ]; then
	  "C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean $msbuildparams
	fi

	rm -f lib/gpumisc/release/gpumisc.lib
	touch src/sawe/configuration/configuration.cpp

	"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" $msbuildparams
	cp src/release/sonicawe.exe src/release/sonicawe-cuda.exe
else
    echo "Couldn't find nvcc, skipping build of \'${packagename}-cuda\'.";
	rm -f src/release/sonicawe-cuda.exe
fi

echo "========================== Building ==========================="
echo "Building Sonic AWE ${packagename} Launcher"

(
	cd dist/package-win/launcher
	qmake "DEFINES+=PACKAGE=\"${packagename}\""
	"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //t:Clean //p:Configuration=Release launcher.sln
	"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //p:Configuration=Release launcher.sln
) || false


echo "========================== Installer =========================="
echo "Creating Windows installer file: $(pwd)/tmp/$filename for package $packagefullname"
rm -rf tmp/$filename
rm -rf $packagefullname
mkdir -p tmp
cp -r lib/sonicawe-winlib/sonicawe_snapshot_win32_base $packagefullname
cp src/release/sonicawe-cpu.exe "$packagefullname/${packagename}-cpu.exe"
[ -e src/release/sonicawe-cuda.exe ] && cp src/release/sonicawe-cuda.exe "$packagefullname/${packagename}-cuda.exe"
cp dist/package-win/launcher/release/launcher.exe "$packagefullname/${packagename}.exe"
cp -r matlab $packagefullname/matlab
cp license/$licensefile $packagefullname
cp dist/package-win/awe_256.ico $packagefullname


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

sed="sed -i"

# append \ to paths for ${File}
$sed -r "s/(^\\$\{File\}.*) (.*$)/\1\\\\ \2/g" $nsisscript

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
#compile & move installer
makensis $nsisscript
mv dist/package-win/$filename tmp/$filename

#clean sonicawe.nsi
rm -f $nsisscript


echo "installer compiled"

popd
