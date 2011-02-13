#!/bin/bash
set -e

version=$(date +0.%Y.%m.%d)
branch=$(git rev-parse --abbrev-ref HEAD)

echo "======================== Deploying Sonic AWE ========================"
echo "branch: ${branch}"
echo "version: ${version}"

read -s -p "Enter password for ftp.sonicawe.com: " pass; echo
if [ -z "$pass" ]; then echo "Missing password for ftp.sonicawe.com, can't deploy."; exit 1; fi
echo "======================== Checking local repo status ========================"
cd ../../gpumisc
if [ -n "$(git status -uno --porcelain)" ]; then echo "In gpumisc: local git repo is not clean."; exit 1; fi
cd ../sonicawe
if [ -n "$(git status -uno --porcelain)" ]; then echo "In sonicawe: local git repo is not clean."; exit 1; fi

echo "======================== Updating local repos ========================"
git pull --rebase origin $branch

cd ../gpumisc
git pull --rebase origin master
