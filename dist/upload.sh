#!/bin/bash
set -e

if [ -z "${version}" ]; then echo "Missing version, can't upload."; exit 1; fi
if [ -z "${filename}" ]; then echo "Missing filename, can't upload."; exit 1; fi
if [ -z "$pass" ]; then echo "Missing password, skipping upload."; return; fi

curl -T "$filename" -u sonicawe.com:$pass --ftp-create-dirs ftp://ftp.sonicawe.com/data/$version/

echo "Uploaded file to:"
url="http://data.sonicawe.com/${version}/${filename}"
echo $url
