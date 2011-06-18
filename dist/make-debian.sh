#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  qmake $qmaketarget
  make distclean
  qmake $qmaketarget
else
  rm -f sonicawe/sonicawe
fi
make -j5

echo "========================== Packaging =========================="
filename="sonicawe_${versiontag}_$(uname -m).deb"
echo "Creating debian archive: $filename version ${version}"
cd sonicawe/dist
source ./package-debian.sh ${versiontag} ${version}

passiveftp=passive
