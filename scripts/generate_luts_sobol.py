#!/usr/bin/env python3

# This scripts generate direction vectors for a sobol sequence up to 21201 dimensions. 
# It is based on Joe & Kuo Sobol sequence generators : https://web.maths.unsw.edu.au/~fkuo/sobol/
# File is divided in columns:   
# d = Dimension of the sobol sequence
# s = Degree of the primitive polynomial 
# a = Coefficients of the primitive polynomials in binary  
# m_i = initial recursive seed of the direction numbers
# ex : 
# d       s       a       m_i     
# 2       1       0       1 
# 3       2       1       1 3 

import copy
import argparse
import subprocess
from numpy import array


def read_as_string(joe_kuo_file_path,num_lines): 
    file = open(joe_kuo_file_path) 
    str = ""
    file.readline()
    index = 0
    for x in file:
        if index >= num_lines - 1:
            break
        str += x
        index += 1
    file.close()
    return str


def to_arraylist(in_str): 
    sep = in_str.splitlines() #list of str
    arrlist = []
    for line in sep:
        splits = line.split(None , 3)
        d = int(splits[0]) #d
        s = int(splits[1]) #s 
        a = int(splits[2]) #a
        m_i = []
        m_i_splits = splits[3].split()
        for d_n in m_i_splits: 
            m_i.append(int(d_n))
        arrlist.append([d , s , a , m_i])
    return arrlist


def coefficient_bit(s , a): 
    bitlist = [] # a's representation in bits
    shift_index = s - 1 
    while shift_index >= 0:
        bitlist.append((a >> shift_index)  & 0x1)
        shift_index -= 1
    return bitlist 

def compute_tail(s , mk_s):
    return (mk_s << s) ^ mk_s


def compute_seed(s , a , m_i): 
    i = 0
    mkj = compute_tail(s , m_i[0])
    while i+1 < s:
        shift = (1 << (s - 1- i)) # 1 to s-1 
        bit = a[s - 1 - i] 
        mkj ^= m_i[i + 1] * bit * shift 
        i += 1
    return mkj 


def compute_at_index(s , a ,  m_i, idx , target):
    if idx == target - 1: 
        return m_i
    elif idx == 0:
        mk = 0
        mk = compute_seed(s , a , m_i)
        m_i.append(mk)
        return compute_at_index(s , a , m_i, idx + 1, target)
    else:
        last_pos = len(m_i) 
        new_mi = m_i[last_pos - s : last_pos]
        mk = compute_seed(s , a , new_mi)
        m_i.append(mk)
        return compute_at_index(s , a , m_i , idx + 1 , target)


def compute_V(s , a , m_i, size_sequence=32):
    a_bits = coefficient_bit(s , a)
    mi_sequence = compute_at_index(s , a_bits , m_i , 0 , size_sequence)[0:size_sequence]
    for i in range(size_sequence): 
        mi_sequence[i] <<= (size_sequence - (i+1)) 
    return mi_sequence


# For testing

def gray_code(i):  
    return i ^ (i >> 1) 


def hash(x , seed):
    x ^= x * 0x3d20adea
    x += seed
    x *= (seed >> 16) | 0x1
    x ^= x * 0x05526c56
    x ^= x * 0x53a22864
    return x


def reverse_bits(x):
    x &= 0xffffffff
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1))
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2))
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4))
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8))
    return ((x >> 16) | (x << 16)) & 0xffffffff


def owen_scramble(x , seed):
    x = reverse_bits(x)
    x = hash(x, seed)
    return reverse_bits(x)


def sobol_sample(row , coef , direction_numbers_size=32 , seed=0xDEADBEEF):
    if coef == 0:
        return 0
    dir_nums = compute_V(row[1] , row[2] , row[3] , direction_numbers_size)
    accum = 0
    gc = gray_code(coef)
    for i in range(0 , direction_numbers_size):
        if (gc >> i) & 0x1:
            accum ^= dir_nums[i]  
    accum = owen_scramble(accum , seed)
    return accum / float(1 << direction_numbers_size) 


def compute_sobol_sequence(row , target , direction_numbers_size = 32):
    ret = [] 
    for i in range(target):
        val = sobol_sample(copy.deepcopy(row) , i , direction_numbers_size)
        ret.append(val)
    return ret 


def write_as_cxx_header(list_rows , num_lines ,sequence_size, output_file):
    cxx_import = "#include<cstdint>\n"
    type_var = "uint32_t "
    if sequence_size == 32: 
        type_var = "uint32_t " 
    elif sequence_size == 16:
        type_var = "uint16_t "
    elif sequence_size == 64: 
        type_var = "uint64_t "
    else: 
        type_var = "uint8_t "
    cxx_var_str = cxx_import + "inline constexpr " + type_var + output_file + "[" + str(num_lines) + "][32] = {\n"
    for i in range(len(list_rows)):
        s = list_rows[i][1] 
        a = list_rows[i][2]
        m_i = list_rows[i][3][:]
        v_list = compute_V(s , a , m_i , sequence_size)
        concat_hex = "{"
        for i in range(len(v_list)): 
            concat_hex += hex(v_list[i]).upper() + ","
        concat_hex = concat_hex[:-1]
        concat_hex += "},\n"
        cxx_var_str += concat_hex
    cxx_var_str += "};"
    ostream = open(output_file + ".h" , "w")
    ostream.write(cxx_var_str)
    ostream.close()


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


def write_as_cxx_bin(list_rows , num_lines, output_file , sequence_size = 32):
    v_list = []
    ostream = open(output_file , "wb")
    for i in range(len(list_rows)):
        s = list_rows[i][1] 
        a = list_rows[i][2]
        m_i = list_rows[i][3][:]
        v_list.append(compute_V(s , a , m_i , sequence_size))
    for i in range(len(v_list)): 
        ostream.write(array(v_list[i]))
    ostream.close()
    generate_object_file(output_file)





parser = argparse.ArgumentParser(description="Generates Sobol direction vectors.")
parser.add_argument('-d', type=int,  required=True,  help='Desired number of dimensions generated.')
parser.add_argument('-s', type=int,  required=True,  help='Size of a direction vector in bits.')
parser.add_argument('-o', type=str,  required=False,  help='Variable name.')
parser.add_argument('-i', type=str,  required=True,  help='File containing sets of primitive polynormials and direction numbers, See: https://web.maths.unsw.edu.au/~fkuo/sobol/')


args = parser.parse_args()

num_lines = args.d
sequence_size = args.s 
jk_file = args.i 
output_file = args.o 

str_rows = read_as_string(jk_file, num_lines)

# list containing the numerical data of the generator in this form : 
# [d , s , a , [m_i1 , m_i2 , ...]] 
list_rows = to_arraylist(str_rows)


write_as_cxx_header(list_rows , num_lines , sequence_size , output_file)
write_as_cxx_bin(list_rows , num_lines, output_file , sequence_size)








