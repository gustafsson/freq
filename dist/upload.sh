#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can upload."; exit 1; fi
if [ -z "${filename}" ]; then echo "Missing filename, can upload."; exit 1; fi

echo "======================== Uploading to ftp ========================"
du -h $filename
echo "Connecting..."
echo "user sonicawe.com $pass
cd data
mkdir $version
cd $version
put $filename" | ftp -n -v ftp.sonicawe.com
echo "Uploaded file to:"
echo "http://data.sonicawe.com/${version}/${filename}"
