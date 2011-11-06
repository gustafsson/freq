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
if [ "Y" == "${verifyRepos}" ] || [ "y" == "${verifyRepos}" ]; then
	verifyRepos=;
fi
if [ -z "${verifyRepos}" ]; then
	cd ../../gpumisc
	if [ -n "$(git status -uno --porcelain)" ]; then echo "In gpumisc: local git repo is not clean."; exit 1; fi
	cd ../sonicawe
	if [ -n "$(git status -uno --porcelain)" ]; then echo "In sonicawe: local git repo is not clean."; exit 1; fi
	cd dist
fi

if [ -z "${rebuildall}" ]; then read -p "Rebuild all code? (Y/n) " rebuildall; echo; fi


read -s -p "Enter password for ftp.sonicawe.com: " pass; echo
if [ -z "$pass" ]; then echo "Missing password for ftp.sonicawe.com, can't deploy."; exit 1; fi

if [ -z "${verifyRepos}" ]; then
	echo "==================== Updating local repos ====================="
	cd ../../gpumisc
	git pull --rebase origin master
	git fetch origin

	cd ../sonicawe
	git pull --rebase origin $branch
	git fetch origin

	cd dist
fi
