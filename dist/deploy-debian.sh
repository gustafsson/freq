#!/bin/bash
set -e

read -s -p "Enter password for ftp.sonicawe.com: " pass; echo
if [ -z "$pass" ]; then echo "Missing password for ftp.sonicawe.com, can't deploy."; exit; fi
cd ../../gpumisc
if [ -n "$(git status -uno --porcelain)" ]; then echo "In gpumisc: local git repo is not clean."; exit; fi
cd ../sonicawe
if [ -n "$(git status -uno --porcelain)" ]; then echo "In sonicawe: local git repo is not clean."; exit; fi

version=$(date +0.%Y.%m.%d)
filename="sonicawe_${version}_snapshot_x86_64.deb"
branch=$(git rev-parse --abbrev-ref HEAD)

git pull --rebase origin $branch
vim dist/package-debian/DEBIAN/control

cd ../gpumisc
git pull --rebase origin master
cd ..

make distclean
qmake
make -j5

cd sonicawe/dist
./package-debian.sh ${version}_snapshot

echo "user sonicawe.com $pass
cd data
mkdir $version
cd $version
passive
put $filename" | ftp -n -v ftp.sonicawe.com > ftplog.log
echo "Uploaded file to:"
echo "http://data.sonicawe.com/${version}/${filename}"

