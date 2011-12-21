#!/bin/bash

if [ "`pwd | grep 'sonic/sonicawe/tests$'`" == "" ]; then
  echo "Run this script from sonic/sonicawe/tests"
  exit
fi

startdir=`pwd`
dirs=`ls -R | tr -d : | grep ^./ | sed 's/^.\// /'`
failed=
success=
testtimestamp=`date --rfc-3339=seconds | sed "s/ /_/" | sed "s/://g"`
logdir="${startdir}/log/${testtimestamp}"
rm -f *_failed.log
mkdir -p ${logdir}


configurations="usecuda nocuda"
for configname in $configurations; do

  # build sonicawe as testlib
  pushd ../..
  pwd
  touch sonicawe/sawe/configuration/configuration.cpp
  rm -f gpumisc/libgpumisc.a
  rm -f {gpumisc,sonicawe}/Makefile
  qmake CONFIG+=testlib CONFIG+=gcc-4.3 CONFIG+=${configname}
  make -j2
  popd


  for name in $dirs; do
    cd "$name"

    if [ -f *.pro ]; then
      echo Entering "$name"

      testname=`echo $name | sed 's/.*\///'`

      rm -f Makefile
      qmake CONFIG+=gcc-4.3
      make 

      ./$testname > ${logdir}/${testname}.log

	  if [ 0 -ne $? ]; then
        rm -f ${startdir}/${testname}_failed.log
        ln -s ${logdir}/${testname}.log ${startdir}/${testname}_failed.log
        failed="${failed}${name} ${configname}\n"
      else
        success="${success}${name} ${configname}\n"
      fi

      echo Leaving $name
    fi

    cd "$startdir"
  done

done # for cudaconfig

echo
echo
echo Succeeded tests:
echo -e $success | sort

echo
echo
echo Failed tests:
echo -e $failed | sort
echo

