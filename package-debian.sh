#!/bin/bash
rm -f package-debian/DEBIAN/*~ && \
cp sonicawe package-debian/usr/bin && \
cp sonicawe.1 package-debian/usr/share/man/man1 && \
pushd package-debian && \
gzip -f usr/share/man/man1/sonicawe.1 && \
rm -f DEBIAN/md5sums && \
for i in `find -name *~`; do rm $i; done && \
for i in `find usr -type f`; do md5sum $i >> DEBIAN/md5sums; done && \
for i in `find usr -type l`; do md5sum $i >> DEBIAN/md5sums; done && \
popd && \
dpkg -b package-debian sonicawe_0.6.25_amd64.deb
