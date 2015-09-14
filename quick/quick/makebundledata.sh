#!/bin/bash

set -e
flatten="-background #f1f3ff -flatten"
mkdir -p iOS_BundleData
src=../f.png

cd iOS_BundleData
rm -f *.png
s=57
convert $src $flatten -gravity center -resize ${s}x${s} -extent ${s}x${s} Icon.png
convert $src $flatten -resize 200% -gravity center -resize $(($s*2))x$(($s*2)) -extent $(($s*2))x$(($s*2)) Icon@2x.png
for s in 60 72 76; do
	convert $src $flatten -gravity center -resize ${s}x${s} -extent ${s}x${s} Icon-${s}.png
	convert $src $flatten -resize 200% -gravity center -resize $(($s*2))x$(($s*2)) -extent $(($s*2))x$(($s*2)) Icon-${s}@2x.png
done
s=29
convert $src $flatten -gravity center -resize ${s}x${s} -extent ${s}x${s} Icon-Small.png
convert $src $flatten -resize 200% -gravity center -resize $(($s*2))x$(($s*2)) -extent $(($s*2))x$(($s*2)) Icon-Small@2x.png
for s in 40 50; do
	convert $src $flatten -gravity center -resize ${s}x${s} -extent ${s}x${s} Icon-${s}.png
	convert $src $flatten -resize 200% -gravity center -resize $(($s*2))x$(($s*2)) -extent $(($s*2))x$(($s*2)) Icon-${s}@2x.png
done

convert ../freq.png $flatten -gravity center -resize 320x480\> -extent 320x480 Default.png
convert ../freq.png $flatten -gravity center -resize 640x960\> -extent 640x960 Default@2x.png
convert ../freq.png $flatten -gravity center -resize 640x1136\> -extent 640x1136 Default-568h@2x.png
convert ../freq.png $flatten -gravity center -resize 768x1004\> -extent 768x1004 Default-Portrait.png
convert ../freq.png $flatten -resize 200% -gravity center -resize 1536x2008\> -extent 1536x2008 Default-Portrait@2x~ipad.png
convert ../freq.png $flatten -gravity center -resize 1024x748\> -extent 1024x748 Default-Landscape.png
convert ../freq.png $flatten -resize 200%  -gravity center -resize 2048x1496\> -extent 2048x1496 Default-Landscape@2x~ipad.png
convert ../freq.png $flatten -gravity center -resize 768x1024\> -extent 768x1024 Default-Portrait-1024h.png
convert ../freq.png $flatten -resize 200%  -gravity center -resize 1536x2048\> -extent 1536x2048 Default-Portrait-1024h@2x~ipad.png
convert ../freq.png $flatten -gravity center -resize 1024x768\> -extent 1024x768 Default-Landscape-768h.png
convert ../freq.png $flatten -resize 200%  -gravity center -resize 2048x1536\> -extent 2048x1536 Default-Landscape-768h@2x~ipad.png
