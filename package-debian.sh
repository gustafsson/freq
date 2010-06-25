#!/bin/bash
rm -f package-debian/DEBIAN/*~ && \
cp sonicawe package-debian/usr/bin && \
cp sonicawe.1 package-debian/usr/share/man/man1 && \
cp sawe_extract_cwt.m package-debian/usr/share/sonicawe && \
cp sawe_extract_cwt_time.m package-debian/usr/share/sonicawe && \
cp sawe_filewatcher.m package-debian/usr/share/sonicawe && \
cp sawe_filewatcher_oct.m package-debian/usr/share/sonicawe && \
cp matlabfilter.m package-debian/usr/share/sonicawe && \
cp matlaboperation.m package-debian/usr/share/sonicawe && \
cp sawe_loadbuffer.m package-debian/usr/share/sonicawe && \
cp sawe_loadbuffer_oct.m package-debian/usr/share/sonicawe && \
cp sawe_loadchunk.m package-debian/usr/share/sonicawe && \
cp sawe_loadchunk_oct.m package-debian/usr/share/sonicawe && \
cp sawe_savebuffer.m package-debian/usr/share/sonicawe && \
cp sawe_savebuffer_oct.m package-debian/usr/share/sonicawe && \
cp sawe_savechunk.m package-debian/usr/share/sonicawe && \
cp sawe_savechunk_oct.m package-debian/usr/share/sonicawe && \
cp credits.txt package-debian/usr/share/sonicawe && \
pushd package-debian && \
gzip -f usr/share/man/man1/sonicawe.1 && \
rm -f DEBIAN/md5sums && \
for i in `find -name *~`; do rm $i; done && \
for i in `find usr -type f`; do md5sum $i >> DEBIAN/md5sums; done && \
for i in `find usr -type l`; do md5sum $i >> DEBIAN/md5sums; done && \
popd && \
dpkg -b package-debian sonicawe_0.6.25_amd64.deb
