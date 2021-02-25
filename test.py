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

base_name = 'weights' + '_' + str(mlp.qtd_in) + 'in_' + str(mlp.qtd_h) + 'h_' + str(mlp.qtd_out) + 'out_'

mlp.load ( base_name )

for sample in test_data:
    erro_aprox, erro_class = mlp.test ( sample[0], sample[1])

    string = f"{erroAprox:.15f}," 
    string += f"{ erroClass }"
    print(string)