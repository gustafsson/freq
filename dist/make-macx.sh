#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building ${packagename} ${versiontag}"

echo "qmaketarget: $qmaketarget"
qmake $qmaketarget -spec macx-g++ CONFIG+=release

if [ "Y" == "${rebuildall}" ]; then
  make clean
fi

touch sonicawe/sawe/configuration/configuration.cpp
rm -f gpumisc/libgpumisc.a
rm -f {sonicawe,gpumisc}/Makefile

no_cores=`/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | grep "Number Of Cores" | sed "s/.*: //g"`
make -j${no_cores}
cp sonicawe/${packagename} sonicawe/${packagename}org


echo "========================== Building ==========================="
echo "Building ${packagename} cuda ${versiontag}"

qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"
echo "qmaketarget: $qmaketarget"
qmake $qmaketarget -spec macx-g++ CONFIG+=release

if [ "Y" == "${rebuildall}" ]; then
  make clean
fi

touch sonicawe/sawe/configuration/configuration.cpp
rm -f gpumisc/libgpumisc.a
rm -f {sonicawe,gpumisc}/Makefile

make -j${no_cores}

cp sonicawe/${packagename}org sonicawe/${packagename}

echo "========================== Building ==========================="
echo "Building Sonic AWE Launcher"
cd sonicawe/dist
cp -r package-macos package-macos~
cd package-macos~
g++ -c -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -o launcher.o launcher.c
g++ -c -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -o launcher-mac.o launcher-mac.cpp
g++ -framework CoreFoundation -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -o launcher launcher.o launcher-mac.o

echo "========================== Packaging =========================="
filename="${packagename}_${versiontag}_macos_i386.zip"
echo "Creating Mac OS X application: $filename version ${version}"
cd ..
ruby package-macx.rb ${packagename}_${versiontag} macos_i386 ../${packagename}
