#!/bin/bash



if [ -z $1 ]; then
	echo "Must specify a folder"
	exit 0
fi
if [[ $(echo $1|tail -c 1) != "/" ]]; then
	DIR=$1"/"
else
	DIR=$1
fi

ls $DIR -l |wc -l

