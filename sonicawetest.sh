#!/bin/bash

qmake -r
make -j12 || exit 125
cd src
./sonicawe --test
exit $?
