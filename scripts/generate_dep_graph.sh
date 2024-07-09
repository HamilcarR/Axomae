#!/bin/bash

# Run from root of project
mkdir -p ../build_metadata/dependencies 
cd ../build_metadata/dependencies
cmake --graphviz=dependencies.dot ../../Axomae 
dot -Tpng dependencies.dot -o dependencies.png && sxiv dependencies.png
