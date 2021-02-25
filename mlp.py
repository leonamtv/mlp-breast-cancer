#!/usr/bin/env python3

from core.MLP import MLP
from random import shuffle
from core.data_prep.prepare_data import split_proportionally

epochs = 10000
file_path = './data/breast-cancer-wisconsin/wdbc-norm.data'

data = []

with open(file_path, 'r') as file :
    for line in file :
        if '?' not in line :
            full_line    = [ float(item) for item in line.split(',') ]
            input_entry  = full_line[:30] # Capturando atributos e removendo primeira coluna
            output_entry = full_line[30:]  # Capturando saída
            data.append(( input_entry, output_entry ))

shuffle ( data )

train_data, test_data = split_proportionally ( data )

mlp = MLP(30, 4, 1, 0.8)

decreases = 10

print('epoca,aprox,class')

for i in range ( 1, epochs + 1 ) :

    if i % 50 == 0 and decreases > 0 :
        decreases -= 1
        mlp.ni /= 2

    erroAproxEpoca = 0
    erroClassEpoca = 0

    data_training = train_data
    shuffle ( data_training )

    for sample in data_training:
        erro_aprox, erro_class = mlp.treinar ( sample[0], sample[1])
        # erro_aprox, erro_class = mlp.treinar ( sample[0], [ 0 if i == 0.05 else 1 for i in sample[1]])
        erroAproxEpoca += erro_aprox
        erroClassEpoca += erro_class

    string =  f"{i},"
    string += f"{erroAproxEpoca:.15f}," 
    string += f"{ erroClassEpoca }"
    print(string)

