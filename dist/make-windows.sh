#!/bin/bash
set -e

packagename="sonicawe_${versiontag}_win32"
filename="${packagename}.zip"

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${packagename}"  
qmake $qmaketarget
"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //m:2 //t:Rebuild //p:Configuration=Release sonic.sln

echo "========================== Packaging =========================="
echo "Creating zip file: $(pwd)/$filename for package $packagename"
cd ..
rm -rf $filename
rm -rf $packagename
cp -r sonicawe_snapshot_win32_base $packagename
cp sonic/sonicawe/release/sonicawe.exe $packagename
zip -r $filename $packagename
mv $filename sonic/sonicawe/dist
cd sonic/sonicawe/dist
