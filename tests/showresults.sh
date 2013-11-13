#!/bin/bash

set -e

mkdir -p results
outputname=results/results-`git rev-parse --short HEAD`.html
rm -f "$outputname"

echo "<html><body><h2>tests succeeded</h2></body></html>" > "$outputname"

failedlogs=(`ls | egrep -e .*-[a-z]+_failed.log`)
failcount=`ls | egrep -c -e .*-[a-z]+_failed.log`
failedtests=(`ls | sed -n 's/\(.*\)-.*_failed\.log/\1/p'`)

header="<html><body>"
footer="</body></html>"


p=`pwd`/
outputname=$p$outputname

echo $header > "$outputname"

# title
echo "<h2>$failedcount tests failed in folder $(pwd)</h2>" >> "$outputname"

for n in $(seq 0 $(($failcount - 1)))
do
	l=${failedlogs[$n]}
	t=${failedtests[$n]}
	echo $t
	cd $p
	testdirs=(`find "." -type d -regex ".*/[-0-9]*${t}"`)
	for d in $testdirs
	do
		echo " $d"
		cd $p$d
	    echo "<p>$t <a href='file://$p$l'>log</a><br>" >> "$outputname"
		testname=`ls | sed -n 's/\(.*\)-result\.png/\1/p'`
		for k in $testname
		do
			echo " $testname"
		    echo "<table width='100%'><tr><td><img width='100%' src='file://$p$d/$k-gold.png'></td><td><img width='100%' src='file://$p$d/$k-diff.png'></td><td><img width='100%' src='file://$p$d/$k-result.png'></td></tr></table>" >> "$outputname"
    	done
		echo "</p>" >> "$outputname"
	done
done

cd $p

echo $footer >> "$outputname"

open "$outputname"
