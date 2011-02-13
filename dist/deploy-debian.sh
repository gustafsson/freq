#!/bin/bash
set -e

. ./prereq.sh

if [ -z "${version}" ]; then echo "Missing version, can upload."; exit 1; fi

packagename="${version}_snapshot"
filename="sonicawe_${packagename}_x86_64.deb"

vim sonicawe/dist/package-debian/DEBIAN/control

echo "========================"
echo "Building Sonic AWE ${packagename}"
make distclean
qmake
make -j5

echo "========================"
echo "Creating debian archive: $filename"
cd sonicawe/dist
./package-debian.sh ${packagename}

. ./upload.sh
