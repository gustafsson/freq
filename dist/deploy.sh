#!/bin/bash
set -e

if [ -z "$pass" ]; then
  . ./prereq.sh
fi

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi


if [ -z "${target}" ]; then 
  versiontag="${version}${snapshot}"
  qmaketarget=
else
  versiontag="${version}-${target}${snapshot}"
  qmaketarget="CONIFG+=TARGET_${target} DEFINES+=TARGET_${target}"
fi

qmaketarget="${qmaketarget} DEFINES+=\"SONICAWE_VERSION=${versiontag}\""


if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
	platform=windows
elif [ "$(uname -s)" == "Linux" ]; then
    platform=debian
elif [ "$(uname -s)" == "Linux" ]; then
    platform=macx
else
    echo "Don't know how to build Sonic AWE for this platform: $(uname -s).";
fi

. ./make-${platform}.sh
. ./upload.sh
