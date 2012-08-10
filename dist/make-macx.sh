#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ..

echo "========================== Building ==========================="
echo "Building ${packagename} ${versiontag}"

echo "qmaketarget: $qmaketarget"
qmake $qmaketarget -spec macx-g++ CONFIG+=release

if [ "Y" == "${rebuildall}" ]; then
  make clean
fi

touch src/sawe/configuration/configuration.cpp
rm -f lib/gpumisc/libgpumisc.a
rm -f {src,lib/gpumisc}/Makefile

typeset -i no_cores
no_cores=`/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | grep -i "Number Of Cores" | sed "s/.*: //g"`
no_cores=2*$no_cores
if [ $no_cores -eq 8 ]; then
  no_cores=14;
fi

make -j${no_cores}
cp src/${packagename} src/${packagename}org


echo "========================== Building ==========================="
echo "Building ${packagename} cuda ${versiontag}"

if [ -e /usr/local/cuda/bin/nvcc ] && [ -z $NOCUDA ]; then
    qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"
    echo "qmaketarget: $qmaketarget"
    qmake $qmaketarget -spec macx-g++ CONFIG+=release

    if [ "Y" == "${rebuildall}" ]; then
      make clean
    fi

    touch src/sawe/configuration/configuration.cpp
    rm -f lib/gpumisc/libgpumisc.a
    rm -f {src,lib/gpumisc}/Makefile

    make -j${no_cores}
else
    echo "Skipping build of \'${packagename}-cuda\'.";
fi

mv src/${packagename}org src/${packagename}

echo "========================== Building ==========================="
echo "Building Sonic AWE Launcher"
mkdir -p tmp
cd tmp
rm -rf package-macos~
cp -r ../dist/package-macos package-macos~
cd package-macos~
#SYSROOT="-isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -m32 -arch i386"
SYSROOT="-mmacosx-version-min=10.5"
g++ -c $SYSROOT -o launcher.o launcher.cpp
g++ -c $SYSROOT -o launcher-mac.o launcher-mac.cpp
g++ -framework CoreFoundation $SYSROOT -o launcher launcher.o launcher-mac.o

echo "========================== Packaging =========================="
filename="${packagename}_${versiontag}.dmg"
echo "Creating Mac OS X application: $filename version ${version}"
cd ..
ruby ../dist/package-macx.rb ${packagename} ${versiontag} osx ../src/${packagename}
cd ../dist
