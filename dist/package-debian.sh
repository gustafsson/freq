#!/bin/bash
rm -f package-debian/DEBIAN/*~ && \
pushd .. && \
cp sonicawe dist/package-debian/usr/bin && \
cp sonicawe.1 dist/package-debian/usr/share/man/man1 && \
cp matlab/sawe_extract_cwt.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_extract_cwt_time.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_filewatcher.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_filewatcher_oct.m dist/package-debian/usr/share/sonicawe && \
cp matlab/matlabfilter.m dist/package-debian/usr/share/sonicawe && \
cp matlab/matlaboperation.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_loadbuffer.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_loadbuffer_oct.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_loadchunk.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_loadchunk_oct.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_savebuffer.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_savebuffer_oct.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_savechunk.m dist/package-debian/usr/share/sonicawe && \
cp matlab/sawe_savechunk_oct.m dist/package-debian/usr/share/sonicawe && \
cp credits.txt dist/package-debian/usr/share/sonicawe && \
popd && \
pushd package-debian && \
gzip -f usr/share/man/man1/sonicawe.1 && \
rm -f DEBIAN/md5sums && \
for i in `find -name *~`; do rm $i; done && \
for i in `find usr -type f`; do md5sum $i >> DEBIAN/md5sums; done && \
for i in `find usr -type l`; do md5sum $i >> DEBIAN/md5sums; done && \
popd && \
dpkg -b package-debian sonicawe_0.8.26-unstable_amd64.deb
