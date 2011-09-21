#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  make distclean
  qmake $qmaketarget -spec macx-g++
fi
make

echo "========================== Building ==========================="
echo "Building Sonic AWE Launcher"
cd sonicawe/dist
cd package-macos
gcc -framework CoreFoundation -o launcher launcher.c

echo "========================== Packaging =========================="
filename="sonicawe_${packagename}_${versiontag}_macos_i386.zip"
echo "Creating Mac OS X application: $filename version ${version}"
cd ..
ruby package-macx.rb ${packagename}_${versiontag} macos_i386 ../${packagename}


passiveftp=passive
