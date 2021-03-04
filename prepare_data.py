#!/usr/bin/env python3

import os
import random as rnd

from random import choice, shuffle

file_path = './data/breast-cancer-wisconsin/wdbc.data'
norm_file_path = './data/breast-cancer-wisconsin/wdbc-norm-bal.data'
mutation_rate = 0.1

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


data_benigno = []
data_maligno = []

for dt in data :
    if 0.95 in dt[1] :
        data_benigno.append ( dt )  
    if 0.05 in dt[1] :
        data_maligno.append ( dt )  

addicional_mutated_data = []

while len ( addicional_mutated_data ) < abs ( len ( data_benigno ) - len ( data_maligno )) :
    random_choice = choice ( data_benigno )
    print(random_choice)
    new_sample = ()
    input_entry, output_entry = random_choice[0], random_choice[1]
    new_entry = []
    for entry in input_entry :
        ruido = rnd.gauss(0, 0.02)
        signal_definer = rnd.randint (0, 100)
        value = ( entry + ruido ) if signal_definer % 2 == 0 else ( entry - ruido )
        new_entry.append ( value )
    new_sample = ( new_entry, output_entry )
    addicional_mutated_data.append(random_choice)
    

for dt in addicional_mutated_data :
    data_maligno.append ( dt )

data = []

while len ( data_benigno ) >= 0 and len ( data_maligno ) > 0 :
    data.append ( data_benigno.pop ())
    data.append ( data_maligno.pop ())

shuffle ( data )

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