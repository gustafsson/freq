#!/bin/bash
set -e

packagename="sonicawe_${versiontag}_win32"
filename="${packagename}_setup.exe"
nsistemplate="sonic/sonicawe/dist/package-win/Sonicawe_template.nsi"
nsisscript="sonic/sonicawe/dist/package-win/Sonicawe.nsi"
nsiswriter="sonic/sonicawe/dist/package-win/Nsi_Writer.exe"
licensefile="license.txt"

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${packagename}"  
if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
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
echo "Creating Windows installer file: $(pwd)/$filename for package $packagename"
cd ..
rm -rf $filename
rm -rf $packagename
cp -r sonicawe_snapshot_win32_base $packagename
cp sonic/sonicawe/dist/package-win/sonicawe.exe.manifest $packagename
cp sonic/sonicawe/release/sonicawe.exe $packagename
cp -r sonic/sonicawe/matlab $packagename/matlab
cp sonic/sonicawe/license/$licensefile $packagename
cp sonic/sonicawe/dist/package-win/awe_256.ico $packagename

#Executing dxdiag for Nvidia driver version minimum requirement
CMD //C dxdiag //x %CD%\\dxdiag.xml
if [ -f dxdiag.xml ]; then
nvid_version=`sed -e '/DriverVersion/ !d' -e 's!<DriverVersion>\([^<]*\)</DriverVersion>!\~&\~!' dxdiag.xml | awk -F"~" '{print $2}' | cut -f2 -d">" | cut -f1 -d"<"`
else
echo Nvidia driver version could not be read because dxdiag xml file was not found. WARNING, version value is set to \"1.0.0.0\" any version of Nvidia drivers will be recognized as compatible.
nvid_version="1.0.0.0"
fi

#execute NsiWriter.exe to create and fill the Sonicawe.nsi script
nsistemplate=`pwd`\/$nsistemplate
nsistemplate=`echo $nsistemplate | sed 's@\\/c\\/@C:\\\\\\\@'`
nsistemplate=`echo $nsistemplate | sed 's@\\/@\\\\\\\@g'`
nsisscriptwin=`pwd`\/$nsisscript
nsisscriptwin=`echo $nsisscriptwin | sed 's@\\/c\\/@C:\\\\\\\@'`
nsisscriptwin=`echo $nsisscriptwin | sed 's@\\/@\\\\\\\@g'`
instfilepath=`pwd`\/$packagename
instfilepathwin=`echo $instfilepath | sed 's@\\/c\\/@C:\\\\@'`
instfilepathwin=`echo $instfilepathwin | sed 's@\\/@\\\\@g'`
instfilepath=`echo $instfilepath | sed 's@\\/c\\/@C:\\\\\\\@'`
instfilepath=`echo $instfilepath | sed 's@\\/@\\\\\\\@g'`
$nsiswriter "$nsistemplate" "$nsisscriptwin" "$instfilepathwin"

#inserting filename, version and nvidia version number in NSIS script 
sed -i.backup -e "s/\!define SA\_VERSION \".*\"/\!define SA\_VERSION \"${versiontag}\"/" $nsisscript 
sed -i.backup -e "s/\!define NVID\_VERSION \".*\"/\!define NVID\_VERSION \"$nvid_version\"/" $nsisscript 
sed -i.backup -e "s/\!define INST\_FILES \".*\"/\!define INST\_FILES \"$instfilepath\"/" $nsisscript 
sed -i.backup -e "s/\!define FILE\_NAME \".*\"/\!define FILE\_NAME \"$filename\"/" $nsisscript 
licensepath=`pwd`\/$packagename\/license.txt
licensepath=`echo $licensepath | sed 's@\\/c\\/@C:\\\\\\\@'`
licensepath=`echo $licensepath | sed 's@\\/@\\\\\\\@g'`
sed -i.backup -e "s/\!insertmacro MUI\_PAGE\_LICENSE \".*\"/\!insertmacro MUI\_PAGE\_LICENSE \"$licensepath\"/" $nsisscript 


#Check for NSIS
type -P makensis &>/dev/null || { echo "NSIS is not installed.  Aborting installer compilation. Please install Nsis to avoid this error" >&2; exit 1; }

#compile & move installer
makensis $nsisscript
mv sonic/sonicawe/dist/package-win/$filename sonic/sonicawe/dist/$filename

#clean sonicawe.nsi for git consistency
rm $nsisscript

cd sonic/sonicawe/dist


