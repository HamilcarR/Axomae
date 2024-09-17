#!/bin/bash

ROOT="$(pwd)"
echo 'Downloading dependencies ...' 
git submodule update --init --force
echo 'Dependencies downloaded ...' 
if [ -d "$ROOT/vendor/qt" ]; then
	cd $ROOT/vendor/qt || exit
else
	echo 'Error: The QT directory has not been created by submodule update ...'
fi


echo 'Setting up required QT submodules ...' 
./init-repository.pl --force --module-subset=qtbase
