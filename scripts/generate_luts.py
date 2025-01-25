#!usr/bin/env python3

# This script generates trigonometric lookup tables that are embedded in the executable.

import sys
import math
from array import array
import subprocess

# Generates f:[-1 , 1] -> [0 , PI]
def generate_acos(precision , path):
    acos_table = []
    for i in range(0 , precision):
        scaled = (i / precision) * 2 - 1 
        acos_table.append(math.acos(scaled))
         
    file = open(path , "wb")
    arr = array('d' , acos_table)
    file.write(arr)
    file.close()
    return path

PRECISION = sys.argv[1]
ACOS_NAME = sys.argv[2]

acos_filename = generate_acos(int(PRECISION) , ACOS_NAME)
subprocess.run([
    "objcopy",
    "-I", "binary",
    "-O", "elf64-x86-64",
    "-B", "i386:x86-64",
    "--rename-section", ".data=.rodata,alloc,load,readonly,data,contents",
    acos_filename,
    acos_filename + ".o"
])

print(acos_filename)
print(acos_filename + ".o")
