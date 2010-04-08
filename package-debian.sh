#!/bin/bash
rm -f package-debian/DEBIAN/*~ && \
cp sonicawe package-debian/opt/sonicawe && \
pushd package-debian && \
md5sum opt/sonicawe/sonicawe > DEBIAN/md5sums && \
popd && \
dpkg -b package-debian sonicawe_0.4.9_amd64.deb
