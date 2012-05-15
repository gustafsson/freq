#!/bin/bash
set -e

echo "Set a variable named 'target' to build for other targets than the default (see dist/deploy-all-targets.sh)"

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

# Check if the submodules have been initialized
if ! grep submodule .git/config > /dev/null; then
    git submodule update --init lib/gpumisc
    
    if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
        git submodule update --init lib/sonicawe-winlib
    elif [ "$(uname -s)" == "Linux" ]; then
        if [ -z `which colorgcc` ]; then
            echo "Some required and recommended libraries seem to be missing, running apt-get"
            glewpkg=libglew1.6-dev
            if [ -z `apt-cache search $glewdeb` ]; then
                glewpkg=libglew1.5-dev
            fi
            sudo apt-get install libsndfile1-dev portaudio19-dev $glewpkg freeglut3-dev libboost-dev libboost-serialization-dev libqt4-dev qtcreator libhdf5-serial-dev qgit build-essential colorgcc git-gui git-doc
        fi
    elif [ "$(uname -s)" == "Darwin" ]; then
        git submodule update --init lib/sonicawe-maclib
    else
        echo "Don't know how to build Sonic AWE for this platform: $(uname -s).";
    fi
fi

sonicawebranch=`git rev-parse --abbrev-ref HEAD`

cd dist
. ./deploy.sh
cd ..

echo "Running Sonic AWE with sonicawe@${sonicawebranch}"
if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
	(
		cd lib/sonicawe-winlib/sonicawe_snapshot_win32_base
		../../../tmp/$packagename/$packagename.exe
	) || false
elif [ "$(uname -s)" == "Linux" ]; then
    src/sonicawe
elif [ "$(uname -s)" == "Darwin" ]; then
    pushd src
    ruby sandboxsonicawe.rb
    popd
    echo "TODO: Locate the binary and make it work"
fi
