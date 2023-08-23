#!/bin/bash

# Execute this script from the base repository. 

qmake CONFIG+=release CONFIG+=debug Axomae.pro
make -j8
