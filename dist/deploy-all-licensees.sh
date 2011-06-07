#!/bin/bash
set -e

urls=

INPUT=licensees.txt
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }


personal=y
echo "Licensee's:"

OLDIFS=$IFS
# IFS is set to 'newline'
IFS="
"
tmp=`tempfile`
for i in `cat $INPUT`; do
	# IFS is set to 'tab'
	IFS="	"
	echo "$i" > $tmp
	read personalemail personallicensetype personalexpired target < $tmp
	echo "email: '$personalemail'	license type: '$personallicensetype'	license expires: '$personalexpired'	license target: '$target'"
done

IFS="
"
for i in `cat $INPUT`; do
	# IFS is set to 'tab'
	IFS="	"
	echo "$i" > $tmp
	read personalemail personallicensetype personalexpired target < $tmp
	echo "email: '$personalemail'	license type: '$personallicensetype'	license expires: '$personalexpired'	license target: '$target'"

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

