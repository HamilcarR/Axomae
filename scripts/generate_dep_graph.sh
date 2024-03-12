#!/bin/bash

# Run from root of project
mkdir -p build/dependencies 
cd build/dependencies
cmake --graphviz=dependencies.dot ../../ 
dot -Tpng dependencies.dot -o dependencies.png && sxiv dependencies.png
