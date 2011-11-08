#!/bin/bash
set -e

if [ -z "$pass" ]; then
  . ./prereq.sh
fi

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi


if [ -z "${target}" ]; then 
  packagename=sonicawe
  versiontag="${version}${snapshot}"
  qmaketarget=
else
  packagename=sonicawe-${target}
  versiontag="${version}${snapshot}"
  qmaketarget="CONIFG+=TARGET_${target} DEFINES+=TARGET_${target} CONFIG+=customtarget CUSTOMTARGET=$packagename"
fi

qmaketarget="${qmaketarget} DEFINES+=\"SONICAWE_VERSION=${versiontag}\""

if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
    platform=windows
elif [ "$(uname -s)" == "Linux" ]; then
    platform=debian
elif [ "$(uname -s)" == "Darwin" ]; then
    platform=macx
else
    echo "Don't know how to build Sonic AWE for this platform: $(uname -s).";
    platform=unknown
fi

. ./make-${platform}.sh
. ./upload.sh

if [ $platform == "windows" ]; then
	packagename="${packagename}-cuda"
	qmaketarget="${qmaketarget} CONFIG+=usecuda CONFIG+=customtarget CUSTOMTARGET=$packagename"

	. ./make-${platform}.sh
	. ./upload.sh
fi

