#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
qmake $qmaketarget
make distclean
qmake $qmaketarget
make -j5

echo "========================== Packaging =========================="
echo "Creating debian archive: $filename"
cd sonicawe/dist
./package-debian.sh ${versiontag} ${version}
filename="sonicawe_${versiontag}_$(uname -m).deb"

passiveftp=passive
