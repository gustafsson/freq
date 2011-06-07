#!/bin/bash
set -e

if [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] && [ -z "$4" ] && [ "$(basename `pwd`)" == "personal-license" ] ; then
	LicenseeEmail=$1
	LicenseType=$2
	LicenseEnd=$3
elif [ -z "$1" ] && [ "$(basename `pwd`)" == "personal-license" ] ; then
    echo "This script can also be executed with parameters, type '$0 help' for more information."
	read -p "Enter name of licensee (ex: simon.johansson@addiva.se): " LicenseeEmail
	read -p "License type (i.e \"Evaluation\", \"-\"): " LicenseType
	read -p "Enter end of licensee (ex: 2011-12-31): " LicenseEnd
else
	echo "Compiles a personal license"
	echo
	echo "SYNOPSIS"
	echo "    create-personal-license.sh email type until"
	echo
	echo "DESCRIPTIION"
	echo "     'email' is the licensee's email address"
	echo
	echo "     'type' is either \"\" or \"Evaluation\""
	echo
	echo "     'until' is the last valid date of the license"
	echo
	echo "Run this script from the sonic/sonicawe/personal-license directory"
    exit
fi

if [ "$LicenseType" != "-" ]; then
  LicenseText="$LicenseType of "
else
  LicenseText=""
fi
LicenseText="${LicenseText}Sonic AWE licensed to $LicenseeEmail until $LicenseEnd"
if [ ! -z "$LicenseType" ]; then
  ApplicationTitle=$LicenseText
else
  ApplicationTitle="Sonic AWE"
fi

LicenseName="`echo $LicenseeEmail | sed s/@/_at_/`-$LicenseEnd"
echo "License text: \"$LicenseText\""
echo "Application title: \"$ApplicationTitle\""
echo "License filename: \"$LicenseName\""

read -p "License text ok? (Y/n): " LicenseTextOk

if [ "$LicenseTextOk" != "Y" ] && [ "$LicenseTextOk" != "y" ] && [ -n "$LicenseTextOk" ]; then
  echo "License text not ok, aborting"
  $(exit 1)
fi

pushd reader
qmake DEFINES+=LICENSEEMASH='\\\"'`../masher/masher "${LicenseText}"`'\\\"' DEFINES+=TITLEMASH='\\\"'`../masher/masher "${ApplicationTitle}"`'\\\"'
make clean all
popd

