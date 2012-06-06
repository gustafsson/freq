#!/bin/bash
set -e

version=$(date +0.%Y.%m.%d)

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" == "master" ]; then
  snapshot=
elif [ "$branch" == "develop" ]; then
  snapshot="-snapshot"
else
  snapshot="-test-$branch"
fi

echo "===================== Deploying Sonic AWE ====================="
echo "branch: ${branch}"
echo "version: ${version}"
echo "release: sonicawe_${version}${snapshot}"

# Check if git user name has been specified
if [ -z "`git config --global user.name`" ]; then
	echo "git has not yet been configured. Configure now or press ^C"
	read -p "Please enter your name, such as Donald Duck: " username
	read -p "Please enter your email, such as donald@duck.com: " usermail
	git config --global user.name "$username"
	git config --global user.email "$usermail"
fi

# Enforce this (being a bit intrusive here)
git config merge.ff false

read -p "Verify and update repositories? (Y/n) " verifyRepos; echo
if [ "N" == "${verifyRepos}" ] || [ "n" == "${verifyRepos}" ]; then
	verifyRepos=N;
else
	verifyRepos=Y;
fi
if [ "Y" == "${verifyRepos}" ]; then
	if [ -n "$(git status -uno --porcelain)" ]; then
		echo "Local git repo is not clean."
		echo
		echo "Commit your changes or run 'git stash' to temporarily store them in the stash."
		echo "Run 'git submodule update' to bring submodules up-to-date."
		echo
		echo "If you have changed the files in a submodule you must first commit to to that"
		echo "submodule's repo. Then you must also update the reference in the sonicawe repo"
		echo "by 'git add'-ing that submodule and make a new commit in the sonicawe repo."
		echo
		echo "Or to build anyways, run buildandrun.sh again and answer n (no) to the previous"
		echo "question."
		false
	fi
fi

if [ -z "${rebuildall}" ]; then read -p "Rebuild all code? (y/N) " rebuildall; echo; fi
if [ "Y" == "${rebuildall}" ] || [ "y" == "${rebuildall}" ]; then
	rebuildall=Y;
else
	rebuildall=N;

	if [ -z "${rebuildcuda}" ]; then read -p ".cu-files (CUDA kernels) are not rebuilt when included .h-files are changed. \"touch\" all .cu-files? (y/N) " rebuildcuda; echo; fi
	if [ "Y" == "${rebuildcuda}" ] || [ "y" == "${rebuildcuda}" ]; then
		(cd ..; touch `find . -name *.cu`)
	fi
fi


read -s -p "Enter password for ftp.sonicawe.com (leave empty to skip uploading): " pass; echo
expectedpass=d0f085d2cfdee0b2128bf80226f6bee5
if [ -z "$pass" ]; then
    echo "Missing password for ftp.sonicawe.com. Won't upload any data."
elif ( [ "`which md5`" != "" ] && [ $expectedpass != "`echo $pass | md5`" ] ) ||
     ( [ "`which md5sum`" != "" ] && [ $expectedpass != "`echo $pass | md5sum | sed 's/ .*//'`" ] )
then
    echo "Wrong password (leave empty to skip upload)."
    exit
fi

if [ "Y" == "${verifyRepos}" ]; then
	echo "==================== Updating local repos ====================="
	git pull --rebase

	# Make sure the submodules are initialized and points to the correct commit
	(
		cd ..

		# note, if they are already initialized this command would do:
		#git submodule update

		git submodule update --init lib/gpumisc

		if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
			git submodule update --init lib/sonicawe-winlib
		elif [ "$(uname -s)" == "Linux" ]; then
			git submodule update --init lib/sonicawe-ubuntulib

			if [ -z `which colorgcc` ]; then
				# In Ubuntu we're using packages from the Ubuntu repo instead of a specific precompiled set of binaries.
				echo "Some required and recommended libraries seem to be missing, running apt-get"
				glewpkg=libglew1.6-dev
				if [ -z `apt-cache search $glewdeb` ]; then
					glewpkg=libglew1.5-dev
				fi
				sudo apt-get install libsndfile1-dev $glewpkg freeglut3-dev libboost-dev libboost-serialization-dev libqt4-dev qtcreator libhdf5-serial-dev qgit build-essential colorgcc git-gui git-doc
			fi
		elif [ "$(uname -s)" == "Darwin" ]; then
			git submodule update --init lib/sonicawe-maclib
		else
			echo "Don't know how to build Sonic AWE for this platform: $(uname -s).";
		fi
	) || false
fi
