#!/bin/bash

ROOT="$(pwd)"
echo 'Downloading dependencies ...' 
git submodule update --init --force

if [ -d "$ROOT/vendor/qt" ]; then
	cd $ROOT/vendor/qt || exit
else
	echo 'Error: The QT directory has not been created by submodule update ...'
fi
echo 'Dependencies downloaded ...' 

echo 'Setting up required QT submodules ...' 
./init-repository.pl --force --module-subset=qtbase
