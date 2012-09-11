#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ..

qmaketarget="${qmaketarget} DEFINES+=SONICAWE_UNAMEm=`uname -m` DEFINES+=SONICAWE_DISTCODENAME=`lsb_release -c | sed "s/.*:\t//g"`"

echo "========================== Building ==========================="
echo "Building ${packagename} ${versiontag}"

echo "qmaketarget: $qmaketarget"
qmake $qmaketarget

if [ "Y" == "${rebuildall}" ]; then
  make clean
fi

touch src/sawe/configuration/configuration.cpp
rm -f lib/gpumisc/libgpumisc.a
rm -f {src,lib/gpumisc}/Makefile

# We need to create multiple packages that can't depend on packages outside the ubuntu repos. So shared things between our packages need to be duplicated.
LD_RUN_PATH=/usr/share/${packagename}
time make -j`cat /proc/cpuinfo | grep -c processor`


echo "========================== Building ==========================="
echo "Building ${packagename} cuda ${versiontag}"

if [ ! -z "`which nvcc`" ] && [ "Y" == "$buildcuda" ]; then
    qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"

    echo "qmaketarget: $qmaketarget"
    qmake $qmaketarget

    if [ "Y" == "${rebuildall}" ]; then
      make clean
    fi

    touch src/sawe/configuration/configuration.cpp
    rm -f lib/gpumisc/libgpumisc.a
    rm -f {src,lib/gpumisc}/Makefile

    LD_RUN_PATH=/usr/share/${packagename}
    time make -j`cat /proc/cpuinfo | grep -c processor`
else
    rm -f src/${packagename}-cuda
    echo "Couldn't find nvcc, skipping build of '${packagename}-cuda'.";
fi


echo "========================== Packaging =========================="
filename="${packagename}_${versiontag}_$(uname -m).deb"
echo "Creating debian archive: $filename version ${version}"
cd dist
fakeroot ./package-debian.sh ${versiontag} ${version} ${packagename}
