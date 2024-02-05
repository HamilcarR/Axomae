#!/bin/bash

# Execute this script from the base repository. 

arg=$1
qmake Axomae.pro CONFIG+=$arg
make -j8
