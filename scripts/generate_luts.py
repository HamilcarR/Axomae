#!usr/bin/env python3

# This script generates trigonometric lookup tables that are embedded in the executable.

import sys
import math
from array import array
import subprocess

# Generates arc-cosine table f:[-1 , 1] -> [0 , PI]
def generate_acos(precision , path):
    acos_table = []
    for i in range(0 , precision):
        scaled = (i / precision) * 2 - 1 
        acos_table.append(math.acos(scaled))
    file = open(path , "wb")
    arr = array('f' , acos_table)
    file.write(arr)
    file.close()
    return path


def generate_object_file(filename):
    subprocess.run([
    "objcopy",
    "-I", "binary",
    "-O", "elf64-x86-64",
    "-B", "i386:x86-64",
    "--rename-section", ".data=.rodata",
    filename,
    filename + ".o"
    ])
    
    subprocess.run(["rm" , filename])




PRECISION = sys.argv[1]
ACOS_NAME = sys.argv[2]

acos_filename = generate_acos(int(PRECISION) , ACOS_NAME)
generate_object_file(acos_filename)


