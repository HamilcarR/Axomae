#!/bin/bash

# Execute this script in this directory (scripts/) 

cd ..
> TODOlist.txt

for file in sources/*.cpp includes/*.h kernels/*.cu* shaders/*; do
	LINE="$(cat $file | grep -n -e '//TODO' -e '//todo' | tr -d '\t' )"
	if ! [ -z "$LINE" ] ; then 
		echo '' >> TODOlist.txt ; 
		echo "$file:" >> TODOlist.txt ;
		echo '' >> TODOlist.txt ;
		echo "$LINE" >> TODOlist.txt
	fi
done 



