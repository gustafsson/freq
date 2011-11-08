#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
qmake $qmaketarget -spec macx-g++ CONFIG+=release

if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  make distclean
#else
  rm -f sonicawe/${packagename}
fi

qmake $qmaketarget -spec macx-g++ CONFIG+=release
make -j`/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | grep "Number Of Cores" | sed "s/.*: //g"`

echo "========================== Building ==========================="
echo "Building Sonic AWE Launcher"
cd sonicawe/dist
cp -r package-macos package-macos~
cd package-macos~
if [[ $packagename == *cuda* ]]; then
  gcc -framework CoreFoundation -o launcher launcher.c
else
  sed -i "" s/launcher/sonicawe/g Info.plist
fi

echo "========================== Packaging =========================="
filename="${packagename}_${versiontag}_macos_i386.zip"
echo "Creating Mac OS X application: $filename version ${version}"
cd ..
ruby package-macx.rb ${packagename}_${versiontag} macos_i386 ../${packagename}
