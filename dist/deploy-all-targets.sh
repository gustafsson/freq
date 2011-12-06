#!/bin/bash
set -e

# setting a target=myThing here will run 
# qmake CONFIG+=TARGET_myThing DEFINES+=TARGET_myThing
urls=
url=

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
	git push origin :$tagname
fi

git tag $tagname
git push origin $tagname
echo "Created tag '$tagname' (at `git show-ref -s --abbrev $tagname`)"

echo

echo "========================== All urls ==========================="
if [ -z "`echo -e $a`" ] && [ -z "$pass" ]; then
	echo "Didn't upload anything (no ftp password was given)"
elif [ -z "`echo -e $a`" ]; then
	echo "Didn't upload anything (unavailable internet connection? wrong ftp password?)"
else
	echo -e $urls
fi
