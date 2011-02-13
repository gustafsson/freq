#!/bin/bash
set -e

. ./prereq.sh

if [ -z "${version}" ]; then echo "Missing version, can upload."; exit 1; fi

packagename="sonicawe_${version}_snapshot_win32"
filename="${packagename}.zip"

echo "========================"
echo "Building Sonic AWE ${packagename}"
qmake
/c/Windows/Microsoft.NET/Framework/v4.0.30319/MSBuild.exe /t:Rebuild /p:Configuration=Release sonic.sln

echo "========================"
echo "Creating zip file: $filename"
cd ..
rm -rf $filename
rm -rf $packagename
pwd
cp -r sonicawe_snapshot_win32_base $packagename
cp sonic/sonicawe/release/sonicawe.exe $packagename
zip -r $filename $packagename
cd sonic/sonicawe/dist
filename="../../../$filename"

. ./upload.sh
