#!/bin/bash
set -e

# test that build output exists
outputname=libcommon.a
if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
    outputname=release/common.lib
fi

[ -f $outputname ]
