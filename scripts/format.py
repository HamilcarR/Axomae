#!/usr/bin/env python3

from subprocess import call
import glob

print("Formatting files ... ")
source_str = "sources/**/*."
src = glob.glob(source_str + "h", recursive=True)
src += glob.glob(source_str + "cpp", recursive=True)
src += glob.glob(source_str + "cu", recursive=True)
src += glob.glob(source_str + "cuh", recursive=True)
src += glob.glob("tests/*.h") + glob.glob("tests/*.cpp")
for s in src:
    call(['clang-format', '-i', '-style=file', s])
