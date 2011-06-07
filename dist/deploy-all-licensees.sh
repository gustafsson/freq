#!/bin/bash
set -e

urls=

INPUT=licensees.txt
OLDIFS=$IFS

# IFS is set to 'tab'
IFS="	"

echo "Licensee's:"
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
cat $INPUT | while read personalemail personallicensetype personalexpired target
do
	echo "email: '$personalemail'	license type: '$personallicensetype'	license expires: '$personalexpired'	license target: '$target'"
done
IFS=$OLDIFS

personal=y

IFS="	"
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read personalemail personallicensetype personalexpired target < $INPUT
do
	if [ -z "$personalemail" ]; then
		continue;
	fi

	IFS=$OLDIFS
	. ./deploy.sh
	urls="${urls}\n$url"
	rebuildall="n"
	IFS="	"
done
IFS=$OLDIFS


echo "========================== All urls ==========================="
echo -e $urls

