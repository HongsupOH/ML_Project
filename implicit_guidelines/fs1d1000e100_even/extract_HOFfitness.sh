#!/bin/bash
ARG1=$1 # put the .log file name here
ARG2=$2 # put how many runs there were here

CUR_DIR=${PWD##*/}

if test -f $CUR_DIR"_fitness.txt"; then
        rm $CUR_DIR"_fitness.txt"
fi

let END=$ARG2-1
for i in $(seq 0 $END); do
        FILENAME=$ARG1"_"$i"_"$i".log"
        awk '/Fame:/{ORS=",";getline;print $1}' pkl_files/$FILENAME |
	sed 's/^.//;s/.$//' >> $CUR_DIR"_fitness.txt"
	echo ""  >> $CUR_DIR"_fitness.txt"
 done
