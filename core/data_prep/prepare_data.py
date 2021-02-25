import os
import matplotlib.pyplot as plt

from random import shuffle

def split_proportionally ( data, train_proportion=0.8) :
    class_count, dataset = agreggate_by_class ( data )
    train_data = []
    test_data = []
    for count in class_count :
        partial_data = [ entry for entry in dataset if entry[1][-1] == float(count) ]
        train_split = int ( train_proportion * len( partial_data ))
        shuffle ( partial_data )
        for item in partial_data[:train_split] :
            train_data.append ( item )
        for item in partial_data[train_split:] :
            test_data.append ( item )
    shuffle(train_data)
    shuffle(test_data)
    return train_data, test_data


# Uso exclusivo para o massas mamográficas
def agreggate_by_class ( data ) :

    class_count = {}
    
    def sorting ( elem ) :
        return elem[1][-1]
    
    data.sort(key=sorting)

    for entry in data :
        key = str(entry[1][-1])
        if key in class_count :
            class_count[key] += 1
        else :
            class_count[key] = 1

    return class_count, data


def filter_dataset ( file_path, format, normalize=False, visualize_distribution=False ) :

    if not os.path.isfile ( file_path ) :
        raise Exception ( f"Arquivo de dados não encontrado: { file_path }" )

    data = []

    with open(file_path, 'r') as file :
        for line in file :
            if '?' not in line :
                full_line    = [ float(item) for item in line.split(',') ]
                input_entry  = full_line[:format['input_size']]
                output_entry = full_line[(format['input_size']):]
                data.append(( input_entry, output_entry ))

    if normalize :
        max_array = [ 0 for _ in range(format['input_size']) ]

        for sample in data :
            for i, inp in enumerate(sample[0]) :
                if inp > max_array[i] :
                    max_array[i] = inp

        if visualize_distribution :
            inputs = [ [] for _ in range(format['input_size']) ]

        for sample in data :
                sample[0][i] = sample[0][i] / float( max_array[i] )
                if visualize_distribution :
                    inputs[i].append(sample[0][i])

        if visualize_distribution :
            for i, inp in enumerate(inputs) :
                plt.figure()
                plt.title(f'Histograma input {i + 1}')
                plt.hist(x=inp, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
                plt.grid(axis='y')
                plt.show(block=False)

        shuffle(data)       
        return data

    else :
        shuffle(data)
        return data
        

    