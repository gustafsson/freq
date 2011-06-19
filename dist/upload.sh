#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can upload."; exit 1; fi
if [ -z "${filename}" ]; then echo "Missing filename, can upload."; exit 1; fi

echo "======================== Uploading to ftp ========================"
echo "Connecting..."
time (echo "user sonicawe.com $pass
cd data
mkdir $version
cd $version
binary
$passiveftp
put $filename" | ftp -n -v ftp.sonicawe.com)
echo "Uploaded file to:"
url="http://data.sonicawe.com/${version}/${filename}"
echo $url
