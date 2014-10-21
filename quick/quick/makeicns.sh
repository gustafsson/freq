#!/bin/bash
# http://www.imagemagick.org/discourse-server/viewtopic.php?f=2&t=15492&p=54769&hilit=circle+mask#p54769

set -e
flatten="-background #f1f3ff -flatten"
rm -rf freq.iconset
mkdir -p freq.iconset
src=../f.png

cd freq.iconset
for s in 16 32 128 256 512; do
	sr=$(($s/2-1))
	convert $src $flatten -gravity center -resize ${s}x${s} -extent ${s}x${s} \
		\( +clone -threshold -1 -negate -fill white -draw "circle $sr,$sr $sr,0" \) \
		-alpha off -compose copy_opacity -composite	\
		icon_${s}x${s}.png
	sr=$(($s-1))
	s2=$(($s*2))
	convert $src $flatten -resize 200% -gravity center -resize ${s2}x${s2} -extent ${s2}x${s2} \
		\( +clone -threshold -1 -negate -fill white -draw "circle $sr,$sr $sr,0" \) \
		-alpha off -compose copy_opacity -composite	\
		icon_${s}x${s}@2x.png
done
cd ..
iconutil -c icns freq.iconset
rm -rf freq.iconset
