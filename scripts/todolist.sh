#!/bin/bash

# Execute this script in this directory (scripts/) 

cd ..
> TODOlist.txt

for file in sources/*.cpp includes/*.h kernels/*.cu* shaders/*; do
	TODO="$(cat $file | grep -e '//TODO' -e '//todo' | tr -d '\t')" 
	LINE="$(cat $file | grep -n -e '//TODO' -e '//todo' | tr -d '\t' | cut -d ':' -f1)"
	if ! [ -z "$LINE" ] ; then 
		echo "$file:$LINE $TODO" >> TODOlist.txt
	fi
done 



