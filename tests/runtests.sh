#!/bin/bash
set -e

if [ "`pwd | grep 'sonic/sonicawe/tests$'`" == "" ]; then
  echo "Run this script from sonic/sonicawe/tests"
  exit
fi

startdir=`pwd`
dirs=`ls -R | tr -d : | grep ^./ | sed 's/^.\// /'`
failed=
success=
testtimestamp=`date --rfc-3339=seconds | sed "s/ /_/" | sed "s/://g"`
rm -f *_failed.log


configurations="usecuda nocuda"
for configname in $configurations; do
  logdir="${startdir}/log/${testtimestamp}-${configname}"
  mkdir -p ${logdir}

  # build sonicawe as testlib
  echo
  echo "Running tests for configuration '${configname}':"
  build_logname=build-${configname}

  ret=0
  (
    cd ../.. &&
    touch sonicawe/sawe/configuration/configuration.cpp &&
    rm -f gpumisc/libgpumisc.a && 
    rm -f {gpumisc,sonicawe}/Makefile &&
    qmake CONFIG+=testlib CONFIG+=gcc-4.3 CONFIG+=${configname} &&
    make -j2
  ) >& ${logdir}/${build_logname}.log || ret=$?

  if [ 0 -ne $ret ]; then
    ln -s ${logdir}/${build_logname}.log ${build_logname}_failed.log
    echo "X!"
    failed="${failed}${configname}. See ${build_logname}_failed.log\n"
    continue
  fi

  for name in $dirs; do
    cd "$name"

    if [ -f *.pro ]; then
      testname=`echo $name | sed 's/.*\///'`

      rm -f Makefile
      qmake CONFIG+=gcc-4.3 CONFIG+=${configname}

      ret=0
      (
        make && 
        echo "======================" &&
        echo "Running '$testname', config: ${configname}" &&
        echo "======================" &&
        ./$testname
      ) >& ${logdir}/${testname}.log || ret=$?

	  if [ 0 -ne $ret ]; then
        rm -f ${startdir}/${testname}_failed.log
        ln -s ${logdir}/${testname}.log ${startdir}/${testname}_failed.log
        failed="${failed}${name} ${configname}. See ${startdir}/${testname}_failed.log\n"
        echo -n "X"
      else
        success="${success}${name} ${configname}\n"
        echo -n "."
      fi
    fi

    cd "$startdir"
  done

  echo
done # for configname

echo
echo
echo Succeeded tests:
if [ -z "$success" ]; then
  echo No tests succeeded.
else
  echo -e $success | sort
fi

echo
echo
echo Failed tests:
if [ -z "$failed" ]; then
  echo No tests failed.
else
  echo -e $failed | sort
fi

echo
echo Test run timestamp: $testtimestamp

