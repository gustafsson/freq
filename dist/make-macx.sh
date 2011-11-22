#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
qmake $qmaketarget -spec macx-g++ CONFIG+=release

if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  qmake $qmaketarget
  make distclean
else
  rm -f sonicawe/${packagename}
  qmake
fi

qmake $qmaketarget -spec macx-g++ CONFIG+=release
make -j`/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | grep "Number Of Cores" | sed "s/.*: //g"`
cp sonicawe/${packagename} sonicawe/${packagename}org


qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"
if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  qmake $qmaketarget CONFIG+=gcc-4.3
  make distclean
else
  rm -f sonicawe/${packagename}-cuda
  qmake
fi

qmake $qmaketarget -spec macx-g++ CONFIG+=release
make -j`/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | grep "Number Of Cores" | sed "s/.*: //g"`

cp sonicawe/${packagename}org sonicawe/${packagename}

echo "========================== Building ==========================="
echo "Building Sonic AWE Launcher"
cd sonicawe/dist
cp -r package-macos package-macos~
cd package-macos~
gcc -framework CoreFoundation -o launcher launcher.c

echo "========================== Packaging =========================="
filename="${packagename}_${versiontag}_macos_i386.zip"
echo "Creating Mac OS X application: $filename version ${version}"
cd ..
ruby package-macx.rb ${packagename}_${versiontag} macos_i386 ../${packagename}
