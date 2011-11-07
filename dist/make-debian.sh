#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Building ==========================="
echo "Building Sonic AWE ${versiontag}"
qmake $qmaketarget CONFIG+=gcc-4.3

if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  make distclean
else
  rm -f sonicawe/${packagename}
fi

qmake $qmaketarget CONFIG+=gcc-4.3

# We need to create multiple packages that can't depend on packages outside the ubuntu repos. So shared things between our packages need to be duplicated.
LD_RUN_PATH=/usr/share/${packagename}
time make -j`cat /proc/cpuinfo | grep -c processor`

cp sonicawe/${packagename} sonicawe/${packagename}org

qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=${packagename}-cuda"
if [ -z "$rebuildall" ] || [ "${rebuildall}" == "y" ] || [ "${rebuildall}" == "Y" ]; then
  qmake $qmaketarget CONFIG+=gcc-4.3
  make distclean
  qmake $qmaketarget CONFIG+=gcc-4.3
else
  rm -f sonicawe/${packagename}-cuda
  qmake
  qmake $qmaketarget CONFIG+=gcc-4.3
fi

LD_RUN_PATH=/usr/share/${packagename}
time make -j`cat /proc/cpuinfo | grep -c processor`

cp sonicawe/${packagename}org sonicawe/${packagename}

echo "========================== Packaging =========================="
filename="${packagename}_${versiontag}_$(uname -m).deb"
echo "Creating debian archive: $filename version ${version}"
cd sonicawe/dist
fakeroot ./package-debian.sh ${versiontag} ${version} ${packagename}

passiveftp=passive
