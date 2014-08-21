#!/bin/bash

qmake -r
rm -f src/sonicawe
make -j12 || exit 125

mkdir -p tmp/sonicawetest
cd src
HEAD=`git rev-parse HEAD`
./sonicawe --test >& ../tmp/sonicawetest/$HEAD
exit $?
