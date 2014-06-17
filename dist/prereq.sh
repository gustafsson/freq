#!/bin/bash
set -e

version=$(date +0.%Y.%m.%d)

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" == "master" ]; then
  snapshot=
elif [ "$branch" == "develop" ]; then
  snapshot="-snapshot"
else
  snapshot="-$branch"
fi

echo "===================== Deploying Sonic AWE ====================="
echo "branch: ${branch}"
echo "version: ${version}"
echo "release: sonicawe_${version}${snapshot}"
echo
echo "(press enter for default answer)"

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

read -p "Verify and update repositories? [Y/n] " verifyRepos; echo
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

if [ -z "${buildcuda}" ]; then read -p "Build CUDA target? [y/N] " buildcuda; echo; fi
if [ "Y" == "${buildcuda}" ] || [ "y" == "${buildcuda}" ]; then
    buildcuda=Y;
else
    buildcuda=N;
fi

if [ -z "${rebuildall}" ]; then read -p "Rebuild all code? [y/N] " rebuildall; echo; fi
if [ "Y" == "${rebuildall}" ] || [ "y" == "${rebuildall}" ]; then
	rebuildall=Y;
else
	rebuildall=N;

    if [ "Y" == "$buildcuda" ]; then
        if [ -z "${rebuildcuda}" ]; then read -p ".cu-files (CUDA kernels) are not rebuilt when included .h-files are changed. \"touch\" all .cu-files? [y/N] " rebuildcuda; echo; fi
        if [ "Y" == "${rebuildcuda}" ] || [ "y" == "${rebuildcuda}" ]; then
            (cd ..; touch `find . -name *.cu`)
        fi
    fi
fi


read -s -p "Enter password for ftp.sonicawe.com (leave empty to skip upload) []: " pass; echo
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

		if uname -s | grep MINGW32_NT > /dev/null; then
			git submodule update --init lib/sonicawe-winlib
		elif [ "$(uname -s)" == "Linux" ]; then
			git submodule update --init lib/sonicawe-ubuntulib

			if [ -z `which colorgcc` ]; then
				# In Ubuntu we're using packages from the Ubuntu repo instead of a specific precompiled set of binaries.
				echo "Some required and recommended libraries seem to be missing, running apt-get install"
				glewpkg=libglew1.6-dev
				if [ -z "`apt-cache search $glewpkg`" ]; then
					glewpkg=libglew1.5-dev
				fi
				echo "$ sudo apt-get install libsndfile1-dev $glewpkg freeglut3-dev libboost-dev libboost-serialization-dev libqt4-dev qtcreator libhdf5-serial-dev qgit build-essential colorgcc git-gui git-doc curl"
				sudo apt-get install libsndfile1-dev $glewpkg freeglut3-dev libboost-dev libboost-serialization-dev libqt4-dev qtcreator libhdf5-serial-dev qgit build-essential colorgcc git-gui git-doc curl
			fi
		elif [ "$(uname -s)" == "Darwin" ]; then
			#git submodule update --init lib/sonicawe-maclib
			if [ -z `which brew` ] && [ -z `which port` ]; then
				echo "Please install macports or homebrew to install required libraries"
				false
			fi

			if [ -n `which brew` ] && [ ! -f "/usr/local/lib/libsndfile.dylib" ];
				echo "Some required libraries seem to be missing, running brew install"
				echo "and modifying portaudio Formula to use --enable-cxx"
				echo "$ brew tap homebrew/science"
				echo "$ brew install portaudio boost libsndfile hdf5"
				read -p "Press Ctrl-C to abort or any other key to continue... " -n1 -s
				brew tap homebrew/science
				brew install portaudio boost libsndfile hdf5

				# Modify portaudio formula to include --enable-cxx
				if [ -n `brew cat portaudio | grep '--enable-cxx'` ]; then
					sed -i '' 's/"--disable-debug",$/"--disable-debug","--enable-cxx",/' /usr/local/Library/Formula/portaudio.rb
		            brew reinstall portaudio
				fi
			fi

			if [ -n `which port` ] && [ ! -f "/opt/local/lib/libsndfile.dylib" ]; then
				echo "Some required libraries seem to be missing, running port install"
				echo "$ sudo port install portaudio libsndfile hdf5-18 boost tbb"
				read -p "Press Ctrl-C to abort or any other key to continue... " -n1 -s
				sudo port install portaudio libsndfile hdf5-18 boost tbb
			fi
		else
			echo "Don't know how to build Sonic AWE for this platform: $(uname -s).";
			false
		fi
	) || false
fi
