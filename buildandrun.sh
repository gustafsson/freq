#!/bin/bash
set -e

echo "Set a variable named 'target' to build for other targets than the default (see dist/deploy-all-targets.sh)"

cd dist
. ./deploy.sh
cd ..

sonicawebranch=`git rev-parse --abbrev-ref HEAD`

echo "Running Sonic AWE with sonicawe@${sonicawebranch}"
if [ "$(uname -s)" == "MINGW32_NT-6.1" ]; then
	(
		cd tmp/$packagename
		$packagename.exe
	) || false
elif [ "$(uname -s)" == "Linux" ]; then
    (
        cd tmp/package-debian~/opt/muchdifferent/sonicawe
        ./sonicawe
    )
elif [ "$(uname -s)" == "Darwin" ]; then
    pushd src
    ruby sandboxsonicawe.rb
    popd
    echo "TODO: Locate the binary and make it work"
fi
