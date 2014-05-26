#!/bin/bash
#
# Syntax:    ./runtests.sh [egrep params]
#
# Examples:  ./runtests.sh
#            ./runtests.sh common\|opencwt
#            ./runtests.sh -v openproject\|end2end/4
#
# Useful when bisecting like so:
#            git bisect start
#            git bisect good
#            git bisect bad
#            git bisect run ./runtests.sh
#

set -e

if [ `dirname $0` != "." ]; then
  echo "Run from sonicawe root folder"
  false
fi

if [ -d .git/refs/bisect ]; then
  git submodule update
fi

cd tests

touch `find . -name *.cpp` `find . -name *.cu` `find . -name *.pro`

./runtests.sh $*
