#!/bin/bash
set -e

# setting a target=myThing here will run 
# qmake CONFIG+=TARGET_myThing DEFINES+=TARGET_myThing
urls=

unset targets
#targets[${#targets[*]}]="addiva"
#targets[${#targets[*]}]="sd"
#targets[${#targets[*]}]="sss"
targets[${#targets[*]}]="reader"
targets[${#targets[*]}]="" # this is the default build

for target in "${targets[@]}"; do
	. ./deploy.sh
	urls="${urls}\n$url"
done

echo

tagname=$versiontag-$platform
if [ "$(uname -s)" == "Linux" ]; then tagname=${tagname}_`uname -m`; fi

if [ -n "`git tag | grep $tagname`" ]; then
	echo "Tag '$tagname' already exists at `git show-ref -s --abbrev $tagname`. Removing old tag and creating a new one"
	git tag -d $tagname
fi

git tag $tagname
echo "Created tag '$tagname' (at `git show-ref -s --abbrev $tagname`)"

echo

echo "========================== All urls ==========================="
echo -e $urls

