#/bin/bash
cp sonicawe package-debian/opt/sonicawe && \
dpkg -b package-debian sonicawe_0.3.30_amd64.deb
