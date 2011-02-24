#!/bin/bash
set -e

bury_copy() { mkdir -p "`dirname $2`" && cp "$1" "$2"; }

if [ -n "$1" ] && [ -n "$2" ] && [ -z "$3" ] && [ "$(basename `pwd`)" == "dist" ] ; then
        versionnumber=$2
        version=$1
elif [ -n "$1" ] && [ -z "$2" ] && [ "$(basename `pwd`)" == "dist" ] ; then
        version=$1
else
	echo "Creates a Sonic AWE package for Debian linux"
	echo
	echo "SYNOPSIS"
	echo "    package-debian.sh version_string"
	echo
	echo "DESCRIPTIION"
	echo "     'version_string' is on the form"
	echo "         0.2011.01.12"
	echo "         0.2011.01.12-snapshot"
	echo
	echo "     Or use 'def' as version_string to produce default format."
	echo
	echo "Run this script from the sonic/sonicawe/dist directory"
    exit
fi

echo $version

if [ $version == "def" ] ; then
	version="$(date +0.%Y.%m.%d-snapshot)"
fi

package=dist/package-debian~
share=$package/usr/share/sonicawe/.

pushd ..
rm -rf $package
cp -r dist/package-debian $package
if [ -n "${versionnumber}" ]; then
	sed -i "s/Version: .*$/Version: ${versionnumber}/g" $package/DEBIAN/control
fi

mkdir -p $package/usr/lib
mkdir -p $package/usr/bin
cp -r /usr/local/cuda/lib64/libcudart.so* $package/usr/lib/.
cp -r /usr/local/cuda/lib64/libcufft.so* $package/usr/lib/.
cp sonicawe $package/usr/bin/.
bury_copy sonicawe.1 $package/usr/local/share/man/man1/.
mkdir -p $share
cp matlab/sawe_extract_cwt.m $share
cp matlab/sawe_extract_cwt_time.m $share
cp matlab/sawe_filewatcher.m $share
cp matlab/matlabfilter.m $share
cp matlab/matlaboperation_amplify.m $share
cp matlab/matlaboperation_lowpass.m $share
cp matlab/sawe_loadbuffer.m $share
cp matlab/sawe_loadchunk.m $share
cp matlab/sawe_savebuffer.m $share
cp matlab/sawe_savechunk.m $share
cp -r license $share
pushd $package
gzip -f usr/local/share/man/man1/sonicawe.1
rm -f DEBIAN/md5sums
for i in `find -name *~`; do rm $i; done
for i in `find usr -type f`; do md5sum $i >> DEBIAN/md5sums; done
for i in `find usr -type l`; do md5sum $i >> DEBIAN/md5sums; done
popd
output_deb="sonicawe_"$version"_`uname -m`.deb"
#http://www.debian.org/doc/debian-policy/ch-controlfields.html
dpkg -b $package dist/$output_deb
echo "OUTPUT"
echo "    `pwd`/dist/$output_deb"
popd
