#!/bin/bash
set -e

version=$(date +0.%Y.%m.%d)

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" == "master" ]; then
  snapshot=
elif [ "$branch" == "develop" ]; then
  snapshot="-snapshot"
else
  snapshot="-test-$branch"
fi

echo "===================== Deploying Sonic AWE ====================="
echo "branch: ${branch}"
echo "version: ${version}"
echo "release: sonicawe_${version}${snapshot}"

read -p "Verify repositories? (Y/n) " verifyRepos; echo
if [ "N" == "${verifyRepos}" ] || [ "n" == "${verifyRepos}" ]; then
	verifyRepos=N;
else
	verifyRepos=Y;
fi
if [ "Y" == "${verifyRepos}" ]; then
	cd ../../gpumisc
	if [ -n "$(git status -uno --porcelain)" ]; then echo "In gpumisc: local git repo is not clean."; exit 1; fi
	cd ../sonicawe
	if [ -n "$(git status -uno --porcelain)" ]; then echo "In sonicawe: local git repo is not clean."; exit 1; fi
	cd dist
fi

if [ -z "${rebuildall}" ]; then read -p "Rebuild all code? (y/N) " rebuildall; echo; fi
if [ "Y" == "${rebuildall}" ] || [ "y" == "${rebuildall}" ]; then
	rebuildall=Y;
else
	rebuildall=N;

	if [ -z "${rebuildcuda}" ]; then read -p ".cu-files (CUDA kernels) are not rebuilt when included .h-files are changed. \"touch\" all .cu-files? (y/N) " rebuildcuda; echo; fi
	if [ "Y" == "${rebuildcuda}" ] || [ "y" == "${rebuildcuda}" ]; then
		(cd ..; touch `find . -name *.cu`)
	fi
fi


read -s -p "Enter password for ftp.sonicawe.com: " pass; echo
expectedpass=d0f085d2cfdee0b2128bf80226f6bee5
if [ -z "$pass" ]; then
    echo "Missing password for ftp.sonicawe.com. Won't upload any data."
elif ( [ "`which md5`" != "" ] && [ $expectedpass != "`echo $pass | md5`" ] ) ||
     ( [ "`which md5sum`" != "" ] && [ $expectedpass != "`echo $pass | md5sum | sed 's/ .*//'`" ] )
then
    echo "Wrong password (leave empty to skip upload)."
    exit
fi


if [ "Y" == "${verifyRepos}" ]; then
	echo "==================== Updating local repos ====================="
	cd ../../gpumisc
	git pull --rebase

	cd ../sonicawe
	git pull --rebase

	cd dist
fi
