#!/bin/bash
set -e

packagename="sonicawe_${versiontag}_win32"
filename="${packagename}_setup.exe"
nsisscript="sonic/sonicawe/dist/package-win/Sonicawe.nsi"
licensefile="license.txt"

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${packagename}"  
cd gpumisc
qmake $qmaketarget
cd ../sonicawe
qmake $qmaketarget
cd ..
qmake $qmaketarget
"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //m:2 //t:Rebuild //p:Configuration=Release sonic.sln

echo "========================== Installer =========================="
echo "Creating Windows installer file: $(pwd)/$filename for package $packagename"
cd ..
rm -rf $filename
rm -rf $packagename
cp -r sonicawe_snapshot_win32_base $packagename
cp sonic/sonicawe/dist/package-win/sonicawe.exe.manifest $packagename
cp sonic/sonicawe/release/sonicawe.exe $packagename
cp sonic/sonicawe/license/$licensefile $packagename

#retrieving nvidia build version for minimum requirement
windir=`cmd //c echo %WINDIR%`
version=`$windir\\\\sysnative\\\\cmd.exe //c REG QUERY "HKLM\SOFTWARE\NVIDIA Corporation\Installer" //v Version`
version=${version:`expr index "$version" *[^0-9]\.`-2}

#inserting filename, version and nvidia version number in NSIS script
sed -i.backup -e "s/\!define NVID\_VERSION \".*\"/\!define NVID\_VERSION \"$version\"/" $nsisscript 
sed -i.backup -e "s/\!define INST\_FILES \".*\"/\!define INST\_FILES \"$packagename\"/" $nsisscript 
sed -i.backup -e "s/\!define FILE\_NAME \".*\"/\!define FILE\_NAME \"$filename\"/" $nsisscript 
licensepath=`pwd`\/$packagename\/license.txt
licensepath=`echo $licensepath | sed 's@\\/c\\/@C:\\\\\\\@'`
licensepath=`echo $licensepath | sed 's@\\/@\\\\\\\@g'`
sed -i.backup -e "s/\!insertmacro MUI\_PAGE\_LICENSE \".*\"/\!insertmacro MUI\_PAGE\_LICENSE \"$licensepath\"/" $nsisscript 

#Check for NSIS
type -P makensis &>/dev/null || { echo "NSIS is not installed.  Aborting installer compilation. Please install Nsis to avoid this error" >&2; exit 1; }

#compile & move installer
makensis $nsisscript
mv $filename sonic/sonicawe/dist
cd sonic/sonicawe/dist
