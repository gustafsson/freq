#!/bin/bash
set -e

if [ -z "$pass" ]; then
  . ./prereq.sh
fi

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi


if [ -z "${target}" ]; then 
  versiontag="sonicawe_${version}_snapshot"
  qmaketarget=
else
  versiontag="sonicawe_${version}_${target}-snapshot"
  qmaketarget="CONIFG+=TARGET_${target} DEFINES+=TARGET_${target}"
fi


if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
	platform=windows
elif [ "$(uname -s)" == "Linux" ]; then
    platform=ubuntu
else
    echo "Don't know how to build Sonic AWE for this platform: $(uname -s).";
fi

. ./make-${platform}.sh
. ./upload.sh
