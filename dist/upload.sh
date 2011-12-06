#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi
if [ -z "${filename}" ]; then echo "Missing filename, can't upload."; exit 1; fi
if [ -z "$pass" ]; then echo "Missing password, skipping upload."; return; fi

echo "======================== Uploading to ftp ========================"
echo "Connecting..."
time (echo "user sonicawe.com $pass
cd data
mkdir $version
cd $version
binary
$passiveftp
put $filename" | ftp -n -v ftp.sonicawe.com || return 1 ) || return
echo "Uploaded file to:"
url="http://data.sonicawe.com/${version}/${filename}"
echo $url
