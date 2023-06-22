#! /bin/bash

valgrind --tool=callgrind --dump-instr=yes -v --instr-atstart=no ./$1 
