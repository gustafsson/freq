#!/bin/bash
set -e

starttimestamp="`date`"

# setting a target=myThing here will run 
# qmake CONFIG+=TARGET_myThing DEFINES+=TARGET_myThing
urls=
url=

unset targets
#targets[${#targets[*]}]="myThing"
targets[${#targets[*]}]="" # this is the default build

if [ -n "$1" ]; then
	targets="$@"
fi

for target in "${targets[@]}"; do
	. ./deploy.sh $target
	urls="${urls}\n$url"
done

echo

if [ -z "`echo -e $urls`" ] && [ -z "$pass" ]; then
	echo "Didn't upload anything (no ftp password was given)"

elif [ -z "`echo -e $urls`" ]; then
	echo "Didn't upload anything (unavailable internet connection? wrong ftp password?)"

else
	tagname=$versiontag-$platform
	if [ "$(uname -s)" == "Linux" ]; then tagname=${tagname}_`uname -m`; fi

	if [ -n "`git tag | grep $tagname`" ]; then
		echo "Tag '$tagname' already exists at `git show-ref -s --abbrev $tagname`. Removing old tag and creating a new one"
		git tag -d $tagname
		git push origin :$tagname || :
	fi

	git tag $tagname
	git push origin $tagname || :
	echo "Created tag '$tagname' (at `git show-ref -s --abbrev $tagname`)"

	echo

	echo "========================== All urls ==========================="
	echo -e $urls
fi

echo
echo "Deploy started at: ${starttimestamp}"
echo "Deploy ended at:   `date`"
