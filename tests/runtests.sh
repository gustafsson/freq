#!/bin/bash
set -e

configurations="onlycpu usecuda"
defaulttimeout=10

if [ "$1" = "--help" ]; then
  scriptName="${0##*/}"
  echo "Run this script from sonic/sonicawe/tests"
  echo
  echo "${scriptName} searches through the subdirectories for tests,"
  echo "compiles them, and executes the binaries produced. All folders"
  echo "containing a .pro file are interpreted as tests named by their"
  echo "folder."
  echo
  echo "When all binaries have been executed a report is printed of"
  echo "which binaries that succeeded (produced an exit code of 0)"
  echo "and which binaries that failed (exit code not equal to 0)."
  echo
  echo "Each project is built with these configurations: "
  echo "{${configurations}} separatedly."
  echo
  echo "To run just one test or a a few tests you can execute ${0##*/}"
  echo "with wildcards that are passed on to grep to match against"
  echo 'relative-path configuration-name'
  echo
  echo "The outputs from the binaries are stored in a folder 'logs'"
  echo "with timestamps. The output from the last run is also copied"
  echo "to the folder 'logs-latest'."
  echo
  echo "You can specify a timeout different than the default ${defaulttimeout} seconds."
  echo "Create a file named timeoutseconds in the project folder"
  echo "containing the number of seconds to use as timeout as text."
  echo "More specified timeouts can be specified using this pattern:"
  echo "timeoutseconds-{cuda,nocuda}-{windows,debian,macx}"
  echo "Note however that if you want a test to succeed or fail "
  echo "depending on the execution time you should check that within"
  echo "the test. This timeout value is just to abort execution of"
  echo "a test that gets stuck and doesn't quit by itself."
  echo
  echo "${0##*/} interprets the name of the folder for each test as"
  echo "the name of the test. The executable in windows is expected to"
  echo "be testname/release/testname.exe, and testname/testname in"
  echo "ubuntu and osx. Tests without such an executable can instead"
  echo "have a script in testname/testname.sh. If neither is found the"
  echo "test fails."
  exit
fi

if [ "`pwd | grep 'sonic/sonicawe/tests$'`" = "" ]; then
  echo "Run this script from sonic/sonicawe/tests"
  exit
fi

startdir=`pwd`
dirs=`ls -R | tr -d : | grep '^\./' | sed 's/^\.\// /' | sort`
failed=
success=
skipped=0

if [ "$(uname -s)" = "MINGW32_NT-6.1" ]; then
    platform=windows
elif [ "$(uname -s)" = "Linux" ]; then
    platform=debian
elif [ "$(uname -s)" = "Darwin" ]; then
    platform=macx
else
    echo "Don't know how to build Sonic AWE for this platform: $(uname -s).";
    platform=unknown
fi

if [ "$platform" = "windows" ]; then
    timestamp(){ echo `date --iso-8601=second`; }
	linkcmd="cp"
	makecmd='"C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" //verbosity:detailed //p:Configuration=Release $( if [ -f *.sln ]; then echo *.sln; elif [ -f *.vcproj ]; then echo *.vcproj; else echo *.vcxproj; fi )'
	makeonecmd='"C:\Program Files (x86)\Microsoft Visual Studio 9.0\vc\vcpackages\vcbuild.exe" //logcommands //time $( if [ -f *.vcproj ]; then echo *.vcproj; else echo *.vcxproj; fi ) "Release|Win32"'

	# make vcbuild called by msbuild detect changes in headers
	PATH="/c/Program Files (x86)/Microsoft Visual Studio 9.0/Common7/IDE:${PATH}"

	PATH="${PATH}:$(cd ../release; pwd)"
	PATH="${PATH}:$(cd ..; pwd)"
	outputdir="release"
else
    timestamp(){ echo `date --rfc-3339=seconds`; }
	qmakeargs="CONFIG+=gcc-4.3"
	linkcmd="ln -s"
	makecmd="make -j2"
	makeonecmd=$makecmd
	outputdir="."
fi

formatedtimestamp() {
	timestamp | sed "s/ /_/" | sed "s/://g"
}
testtimestamp=`formatedtimestamp`
rm -f *_failed.log

logbasedir="${startdir}/logs/${testtimestamp}"
echo "Sawing logs in $logbasedir for {$configurations}"

for configname in $configurations; do
  logdir="${logbasedir}/${configname}"
  mkdir -p ${logdir}

  # build sonicawe as testlib
  echo
  echo "Building and running tests for configuration '${configname}':"
  build_logname=build-${configname}

  ret=0
  (
    cd ../.. &&
    echo $now &&
    pwd &&

	# need to relink both gpumisc and sonicawe when switching configurations
    touch sonicawe/sawe/configuration/configuration.cpp &&
	rm -f {gpumisc,sonicawe}/Makefile &&
    rm -f gpumisc/libgpumisc.a &&
    rm -f sonicawe/libsonicawe.so &&
    rm -f gpumisc/{debug,release}/gpumisc.lib &&

    qmakecmd="qmake CONFIG+=testlib $qmakeargs CONFIG+=${configname}" &&
    echo $qmakecmd &&
    $qmakecmd &&
    (cd gpumisc && $qmakecmd) &&
    (cd sonicawe && $qmakecmd) &&
    eval echo $makecmd &&
    eval time $makecmd &&
    if [ "$platform" = "windows" ]; then
      ls -l gpumisc/release/gpumisc.lib sonicawe/release/sonicawe.lib
    else
      ls -l gpumisc/libgpumisc.a sonicawe/libsonicawe.so
    fi
  ) >& ${logdir}/${build_logname}.log || ret=$?

  if (( 0 != ret )); then
    $linkcmd ${logdir}/${build_logname}.log ${build_logname}_failed.log
    echo "X!"
    failed="${failed}${configname}\n"

  else

  echo -n "["

  for name in $dirs; do
    if ! [ -f $name/*.pro ] || [ -f $name/notest ]; then
	  continue
	fi

    if [ "" != "$*" ] && [ -z "$( echo "${name} ${configname}" | eval grep $* )" ]; then
	  echo -n "_"
	  skipped=$(( skipped + 1 ))
      continue
    fi

    cd "$name"

    testname=`echo $name | sed 's/.*\///'`
	testname=`basename $name`

    timeout=$defaulttimeout
    if [ -f timeoutseconds ]; then
      timeout=`cat timeoutseconds`;
    fi
    if [ -f timeoutseconds-${configname} ]; then
      timeout=`cat timeoutseconds-${configname}`;
    fi
    if [ -f timeoutseconds-${configname}-${platform} ]; then
      timeout=`cat timeoutseconds-${configname}-${platform}`;
    fi

    ret=0
    (
      echo $now &&
      pwd &&
      rm -f Makefile &&
      rm -f $outputdir/$testname &&
	  echo qmake $qmakeargs CONFIG+=${configname} &&
      qmake $qmakeargs CONFIG+=${configname} &&
      eval echo $makeonecmd &&
      eval time $makeonecmd &&
      (
        [ -f $testname.sh ] && ./$testname.sh
      ) || (
        echo "===============================================================================" &&
        echo "$(timestamp): Running '$testname', config: ${configname}, timeout: ${timeout} s." &&
        echo "===============================================================================" &&
        ls -l $outputdir/$testname &&
        time ${startdir}/timeout3.sh -t ${timeout} $outputdir/$testname &&
        echo "===============================================================================" &&
        echo "$(timestamp): Test '$testname' succeeded." &&
        echo "==============================================================================="
      ) || (
	    exitcode=$?
        echo "==============================================================================="
		if (( 143 == exitcode )); then
          echo "$(timestamp): Test '$testname' failed due to time out after ${timeout} seconds and was killed prior to normal exit."
		else
          echo "$(timestamp): Test '$testname' failed with exit code $exitcode."
		fi
        echo "==============================================================================="
	    exit $exitcode
	  )
	) >& ${logdir}/${testname}.log || ret=$?

    if (( 0 != ret )); then
      rm -f ${startdir}/${testname}_failed.log
      $linkcmd ${logdir}/${testname}.log ${startdir}/${testname}-${configname}_failed.log
      failed="${failed}${name} ${configname}\n"
      echo -n "x"
    else
      success="${success}${name} ${configname}\n"
      echo -n "."
    fi

    cd "$startdir"
  done

  echo "]"
  fi

  latestlog="${startdir}/logs-latest/${configname}"
  rm -rf "${latestlog}"
  mkdir -p "${latestlog}"
  cp -r "${logdir}/" "${latestlog}"

done # for configname

if (( 0 < skipped )); then
  echo
  echo $skipped tests were skipped.
fi

echo
echo
echo $(echo -e $success | grep -c .) tests succeeded:
echo -e $success

echo
echo $(echo -e $failed | grep -c .) tests failed:
echo -e $failed

echo
echo "Test run timestamp: $testtimestamp"
echo "Test finished at:   $(formatedtimestamp)"
