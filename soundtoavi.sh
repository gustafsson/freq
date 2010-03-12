#!/bin/bash
IS64=`if [ -n "\`uname -m | grep x86_64\`" ];then echo 64; fi`
export LD_LIBRARY_PATH=../misc:/usr/local/cuda/lib$IS64
./sonicawe $1 $2 $3 $4 $5 $6 $7 $8 $9 --get_chunk_count=1

N=$?

rm image-*.png

for i in $(seq 0 $N)
do
   echo "========= CHUNK  $i  ============"
   rm sonicawe-1.csv
   ./sonicawe $1 $2 $3 $4 $5 $6 $7 $8 $9 --extract_chunk=$i
   octave plotvideo1.m
done

if [ -f spectra2d-unversioned.avi ]
then
	mv spectra2d-unversioned.avi spectra2d-unversioned.avi~
fi

./encavi.sh

