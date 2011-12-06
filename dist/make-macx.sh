#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
echo "qmaketarget: $qmaketarget"

if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  make distclean
else
  rm -f sonicawe/${packagename}
  rm -f gpumisc/libgpumisc.a
  rm {sonicawe,gpumisc}/Makefile
fi

qmake $qmaketarget -spec macx-g++ CONFIG+=release
no_cores=`/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | grep "Number Of Cores" | sed "s/.*: //g"`
make -j${no_cores}
cp sonicawe/${packagename} sonicawe/${packagename}org


qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"
echo "qmaketarget: $qmaketarget"
if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  make distclean
else
  rm -f sonicawe/${packagename}-cuda
  rm -f gpumisc/libgpumisc.a
  rm {sonicawe,gpumisc}/Makefile
fi

qmake $qmaketarget -spec macx-g++ CONFIG+=release
make -j${no_cores}

cp sonicawe/${packagename}org sonicawe/${packagename}

echo "========================== Building ==========================="
echo "Building Sonic AWE Launcher"
cd sonicawe/dist
cp -r package-macos package-macos~
cd package-macos~
gcc -framework CoreFoundation -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386 -o launcher launcher.c

echo "========================== Packaging =========================="
filename="${packagename}_${versiontag}_macos_i386.zip"
echo "Creating Mac OS X application: $filename version ${version}"
cd ..
ruby package-macx.rb ${packagename}_${versiontag} macos_i386 ../${packagename}
