#!/bin/bash



ROOT="$(pwd)"
echo 'Downloading dependencies ...'

# Initialize dependencies that also need their submodules downloaded as well
for dependency in assimp boost glm googletest imath libdeflate openexr SDL SDL_image stb zlib ; do
  git submodule update --init --recursive vendor/$dependency
done
echo 'Dependencies downloaded ...' 
if [ -d "$ROOT/vendor/qt" ]; then
	cd $ROOT/vendor/qt || exit
else
	echo 'Error: The QT directory has not been created by submodule update ...'
fi


echo 'Setting up required QT submodules ...' 
./init-repository.pl --force --module-subset=qtbase
