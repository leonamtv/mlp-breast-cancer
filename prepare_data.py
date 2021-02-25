#!/usr/bin/env python3

import os

file_path = './data/breast-cancer-wisconsin/wdbc.data'
norm_file_path = './data/breast-cancer-wisconsin/wdbc-norm.data'

if not os.path.isfile ( file_path ) :
    raise Exception ( f"Arquivo de dados não encontrado: { file_path }" )

data = []

with open(file_path, 'r') as file :
    for line in file :
        if '?' not in line :
            full_line    = [ item for item in line.split(',') ]
            input_entry  = [ float ( i ) for i in full_line[2:]]                # Capturando atributos e removendo primeira coluna
            output_entry = [ 0.05 if i == 'M' else 0.95 for i in full_line[1]]  # Capturando saída
            data.append(( input_entry, output_entry ))

max_values = [ -float('inf') for _ in range ( len ( data[0][0] ))]
min_values = [ +float('inf') for _ in range ( len ( data[0][0] ))]

for entry in data :
    input_data, _ = entry
    for i in range ( len ( input_data )) :
        if input_data[i] > max_values[i] :
            max_values[i] = input_data[i]
        if input_data[i] < min_values[i] :
            min_values[i] = input_data[i]

normalized_data = []

for entry in data :
    input_data, output_data = entry
    normalized_input  = []
    for i in range ( len ( input_data )):
        normalized_input.append ( float ( input_data [ i ] / ( max_values [ i ] - min_values [ i ])))
    normalized_data.append (( normalized_input, output_data ))
    
data = normalized_data

output_file = open(norm_file_path, 'a')

for entry in data :
    input_data, output_data = entry
    aux_str = ''
    for i in input_data :
        aux_str += str ( i ) + ','
    for i in output_data :
        aux_str += str ( i ) + '\n'
    output_file.write ( aux_str )

output_file.close()