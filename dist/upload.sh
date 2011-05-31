#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can upload."; exit 1; fi
if [ -z "${filename}" ]; then echo "Missing filename, can upload."; exit 1; fi

echo "======================== Uploading to ftp ========================"
du -h package-win/$filename
echo "Connecting..."
echo "user sonicawe.com $pass
cd data
mkdir $version
cd $version
binary
$passiveftp
put package-win\/$filename" | ftp -n -v ftp.sonicawe.com
echo "Uploaded file to:"
url="http://data.sonicawe.com/${version}/${filename}"
echo $url
