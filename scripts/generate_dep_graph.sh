#!/bin/bash

# Run from root of project
mkdir -p ../build_metadata/dependencies 
cd ../build_metadata/dependencies
cmake -DAXOMAE_FROMSOURCE_QT_BUILD=OFF -DAXOMAE_USE_CUDA=OFF --graphviz=dependencies.dot ../../Axomae 
dot -Tpng dependencies.dot -o dependencies.png
