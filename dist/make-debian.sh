#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi

cd ../..

echo "========================== Settings ==========================="
pushd sonicawe/personal-license
if [ -n "${personalemail}" ]; then
  source create-personal-license.sh ${personalemail} ${personallicensetype} ${personalexpired}
elif [ -z "${personal}" ] || [ "${personal}" == "y" ] || [ "${personal}" == "Y" ]; then
  source create-personal-license.sh
else
  source create-personal-license.sh internal@sonicawe.com - ---
fi
popd

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
filename="sonicawe_${versiontag}_$(uname -m)_${LicenseName}.deb"
echo "Creating debian archive: $filename for $LicenseName ($personalemail), version ${version}"
cd sonicawe/dist
./package-debian.sh ${versiontag} ${version}

passiveftp=passive
