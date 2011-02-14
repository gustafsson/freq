#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

packagename="${version}_snapshot"
filename="sonicawe_${packagename}_$(uname -m).deb"

cd ../..

echo "======================== Building ========================"
echo "Building Sonic AWE ${packagename}"
make distclean
qmake $qmaketarget
make -j5

echo "======================== Packaging ========================"
echo "Creating debian archive: $filename"
cd sonicawe/dist
./package-debian.sh ${packagename} ${version}

passiveftp=passive
