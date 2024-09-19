#!/bin/bash 


AXOMAE_ROOT=$(pwd)
WORK_DIR=$(dirname $(realpath compile_commands.json))
cd $WORK_DIR
ctest
cd $AXOMAE_ROOT
