#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
make distclean
qmake -spec macx-g++
make

echo "========================== Packaging =========================="
echo "Creating Mac OS X application: $filename"
cd sonicawe/dist
ruby package-macx.rb ${versiontag} ${version}
filename="sonicawe_${versiontag}_macos_i386.zip"

passiveftp=passive
