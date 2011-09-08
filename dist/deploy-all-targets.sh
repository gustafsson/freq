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

echo "========================== All urls ==========================="
echo -e $urls

