#!/usr/bin/env python3

import tensorflow as tf

from core.data_prep.prepare_data import filter_dataset

def create_model ( ) :

    input_layer   = tf.keras.layers.Input(shape=(None, 30))
    dense_layer_1 = tf.keras.layers.Dense(3, activation='sigmoid')(input_layer)
    dense_layer_2 = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer_1)

    model = tf.keras.Model(inputs=[ input_layer ], outputs=[ dense_layer_2 ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


epochs = 1000

file_path = './data/breast-cancer-wisconsin/wdbc-norm.data'

data = filter_dataset (file_path, format={ 'input_size' : 30 }, normalize=False)
new_data = []

for entry in data :
    input, output = entry
    new_output = [ 0 if i == 0.05 else 1 for i in output ]
    new_data.append(( input, new_output ))

data = new_data

train_split = int( 0.8 * len(data))

train_data, test_data = data[:train_split], data[train_split:]

def train ( model, train_input, train_label ) :
    model.fit(train_input, train_label, epochs=1000)    

def test ( model, test_input, test_label ) :
    model.evaluate( test_input, test_label )

train_input = [ sample[0] for sample in train_data ]
train_label = [ sample[1] for sample in train_data ]

test_input  = [ sample[0] for sample in test_data ]
test_label  = [ sample[1] for sample in test_data ]

model = create_model()

train( model, train_input, train_label )
test ( model, test_input, test_label )
