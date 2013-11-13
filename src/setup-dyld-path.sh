#!/bin/bash

export DYLD_LIBRARY_PATH=$(cd ../lib/sonicawe-maclib/lib; pwd)

if [ "1" != `ps 2> /dev/null | grep -c $PPID` ]; then
    echo "Make sure to run with 'source setup-dyld-path.sh' or '. setup-dyld-path.sh'"
fi
