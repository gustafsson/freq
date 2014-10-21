#!/bin/bash

set -e
flatten="-background #f1f3ff -flatten"
rm -rf freq.iconset
mkdir -p freq.iconset
src=../f.png

cd freq.iconset
for s in 16 32 128 256 512; do
	convert $src $flatten -gravity center -resize ${s}x${s} -extent ${s}x${s} icon_${s}x${s}.png
	convert $src $flatten -resize 200% -gravity center -resize $(($s*2))x$(($s*2)) -extent $(($s*2))x$(($s*2)) icon_${s}x${s}@2x.png
done
cd ..
iconutil -c icns freq.iconset
rm -rf freq.iconset
