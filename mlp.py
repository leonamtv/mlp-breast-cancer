#!/usr/bin/env python3

from core.MLP import MLP
from random import shuffle
from core.data_prep.prepare_data import split_proportionally, filter_dataset

epochs = 100
file_path = './data/breast-cancer-wisconsin/wdbc-norm.data'

data = filter_dataset ( file_path, format={ 'input_size' : 30 }, normalize=False)

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
        erroAproxEpoca += erro_aprox
        erroClassEpoca += erro_class

    string =  f"{i},"
    string += f"{erroAproxEpoca:.15f}," 
    string += f"{ erroClassEpoca }"
    print(string)

base_name = 'weights' + '_' + str(mlp.qtd_in) + 'in_' + str(mlp.qtd_h) + 'h_' + str(mlp.qtd_out) + 'out_'
mlp.dump ( base_name )
