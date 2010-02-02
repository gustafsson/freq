#!/bin/bash
IS64=`if [ -n "\`uname -m | grep x86_64\`" ];then echo 64; fi`
export LD_LIBRARY_PATH=../misc:/usr/local/cuda/lib$IS64
./sonicawe $1 $2 $3 $4 $5 $6 $7 $8 $9
#report="valgrind `date +"%F %H:%M:%S"`.txt"
#valgrind --leak-check=summary --show-reachable=yes --track-origins=yes -v ./sonicawe $1 $2 $3 $4 $5 $6 $7 $8 $9 >& "$report"
#gedit "$report" &
