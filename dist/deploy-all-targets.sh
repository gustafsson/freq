#!/bin/bash
set -e

read -p "Rebuild all code? (Y/n) " simpleclean; echo
if [ "${simpleclean}" == "y" ] || [ "${simpleclean}" == "Y" ]; then
simpleclean=
fi

# setting a target=myThing here will run 
# qmake CONFIG+=TARGET_myThing DEFINES+=TARGET_myThing
urls=

target=sss
. ./deploy.sh; urls="${urls}\n$url"

#target=addiva
#. ./deploy.sh; urls="${urls}\n$url"

target=sd
. ./deploy.sh; urls="${urls}\n$url"

target=
. ./deploy.sh; urls="${urls}\n$url"

echo "========================== All urls ==========================="
echo -e $urls
