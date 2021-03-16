#!/usr/bin/env python3

from core.MLP import MLP
from random import shuffle
from core.data_prep.prepare_data import split_proportionally, filter_dataset

file_path = './data/breast-cancer-wisconsin/wdbc-norm.data'

data = filter_dataset ( file_path, format={ 'input_size' : 30 }, normalize=False)
shuffle ( data )
train_data, test_data = split_proportionally ( data, 0.8 )

mlp = MLP(30, 15, 1, 0.8)

epochs = 200
PLOT = False
decreases = 10
variable_ni = True

erros_class_graf = []
erros_aprox_graf = []
erros_class_graf_teste = []
erros_aprox_graf_teste = []

if not PLOT :
    print("Train dataset size: " + str(len(train_data)))
    print("Test dataset size.: " + str(len(test_data)))

plot_content = 'erroClassGraf,erroAproxGraf,erroClassGrafTeste,erroAproxGrafTeste\n'

if not PLOT :
    print('epoca,aprox,class,taprox,tclass')

for i in range ( 1, epochs + 1 ) :

    if i % 50 == 0 and decreases > 0 and variable_ni:
        decreases -= 1
        mlp.ni /= 2

    erroAproxEpoca = 0
    erroClassEpoca = 0

    data_training = train_data
    shuffle ( data_training )

    for sample in data_training:
        erro_aprox, erro_class = mlp.treinar ( sample[0], sample[1] )
        erroAproxEpoca += erro_aprox
        erroClassEpoca += erro_class

    erroGrafClass = erroClassEpoca / len ( train_data )
    erros_class_graf.append(erroGrafClass)
    erros_aprox_graf.append(erroAproxEpoca)

    if not PLOT :
        string =  f"{i}, "
        string += f"{erroAproxEpoca:.15f}, " 
        string += f"{ erroClassEpoca }, "
        print(string, end='')

    erroAproxEpocaTeste = 0
    erroClassEpocaTeste = 0

    for sample in test_data:
        erro_aprox, erro_class = mlp.test ( sample[0], sample[1])
        erroAproxEpocaTeste += erro_aprox
        erroClassEpocaTeste += erro_class

    erroGrafClassTeste = erroClassEpocaTeste / len ( test_data )
    erros_class_graf_teste.append(erroGrafClassTeste)
    erros_aprox_graf_teste.append(erroAproxEpocaTeste)

    if not PLOT :
        string = f"{erroAproxEpocaTeste:.15f}, " 
        string += f"{erroClassEpocaTeste}"
        print ( string )

# base_name = 'weights' + '_' + str(mlp.qtd_in) + 'in_' + str(mlp.qtd_h) + 'h_' + str(mlp.qtd_out) + 'out_'
# mlp.dump ( base_name )

if PLOT :
    max_err_app = max(erros_aprox_graf)
    max_err_app_teste = max(erros_aprox_graf_teste)

    for ecg, eag, ecgt, eagt in zip ( erros_class_graf, erros_aprox_graf, erros_class_graf_teste, erros_aprox_graf_teste ) :
        plot_content += f"{ecg:.15f},{(eag / max_err_app):.15f},{ecgt:.15f},{(eagt / max_err_app_teste):.15f}\n"

    print(plot_content)
