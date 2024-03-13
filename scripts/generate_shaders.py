#!usr/bin/env python3

# This script generates shaders in a format that we can embed in the executable

import sys
import glob

MAX_LINE = 12


def create_cxx_vars(filename):
    file_name = filename.split('/')[-1]
    file_name = file_name.replace(".", "_")
    str_variable = "glsl_" + file_name
    size_variable = str_variable + "_len"
    return {'str_variable': str_variable, 'size_variable': size_variable}


def generate_str_hex(file_content):
    b_array = bytearray(file_content.encode())
    str_hex = "{\n"
    i = 1
    for h in b_array:
        hex_string = "0x" + '{:02x}'.format(h)
        if i % MAX_LINE == 0:
            str_hex = str_hex + hex_string + ",\n"
        else:
            str_hex = str_hex + hex_string + ", "
        i = i + 1
    str_hex = str_hex[:-2]
    str_hex += "};"
    return {'size': i - 1, 'str': str_hex}


def shader_to_str(path):
    str_vars = create_cxx_vars(path)
    file_str = open(path)
    decl_shader_code = "unsigned char " + str_vars['str_variable'] + "[] = "
    decl_shader_size = "unsigned int " + str_vars['size_variable'] + " = "
    str_struc = generate_str_hex(file_str.read())
    final_shader_str = decl_shader_code + str_struc['str'] + "\n" + decl_shader_size + str(str_struc['size']) + ";\n"
    file_str.close()
    return final_shader_str


SHADER_DIR = sys.argv[1]
GEN_DIR = sys.argv[2]

vertex_files = glob.glob(SHADER_DIR + "/*.vert")
fragment_files = glob.glob(SHADER_DIR + "/*.frag")

vert_new_str = ""
for file in vertex_files:
    vert_new_str = vert_new_str + shader_to_str(file) + "\n"

frag_new_str = ""
for file in fragment_files:
    frag_new_str = frag_new_str + shader_to_str(file) + "\n"

v_file = open(GEN_DIR + "/vertex.h", "w")
v_file.write(vert_new_str)
v_file.close()

f_file = open(GEN_DIR + "/fragment.h", "w")
f_file.write(frag_new_str)
f_file.close()
