#!/bin/bash

# Useful when bisecting like so:
# git bisect start
# git bisect good
# git bisect bad
# git bisect run ./runtests.sh

if [ `dirname $0` != "." ]; then
  echo "Run from sonicawe root folder"
  false
fi

if [ -d .git/refs/bisect ]; then
  git submodule update
fi

cd ~/dev/sonicawe/tests
touch `find . -name *.cpp` `find . -name *.cu` `find . -name *.pro`
./runtests-untracked.sh $*
r=$?

echo "Trying to show results ... "
./showresults.sh

echo "Exiting with error code $r"
exit $r
