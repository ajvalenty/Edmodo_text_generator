#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:17:04 2018

@author: Valenty
"""
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

class writing(object):
    def __init__(self, name, filename):
        self.filename = filename    
        self.name = name
        
    def __getname__(name):
        return name;
    def __getfilename__(filename):
        return filename;
    def database(filename):
        name = filename + ".txt"
        return name
    def execute(object):
        nme = "edmodo.txt"
        raw = open(nme).read()
        raw = raw.lower()
        
        chars = sorted(list(set(raw)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i, c) for i, c in enumerate(chars))
      
        otherchars = len(raw)
        vocab = len(chars)
        print("Total Characters: ", otherchars)
        print("Total Vocab: ", vocab)
        

        length = 3000
        Xdata = []
        Ydata = []
        for i in range(0, otherchars - length, 1):
            seq_in = raw[i:i + length]
            seq_out = raw[i + length]
            Xdata.append([char_to_int[char] for char in seq_in])
            Ydata.append(char_to_int[seq_out])
        patterns = len(Xdata)
        print("Total Patterns: ", patterns)
    
        X = numpy.reshape(Xdata, (patterns, length, 1))
        X = X / float(vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(Ydata)
   
        model = Sequential()
        model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        filename = "edmodo-weights.hdf5"
        model.load_weights(filename)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        start = numpy.random.randint(0, len(Xdata)-1)
        pattern = Xdata[start]
        print("Seed:")
        print(''.join([int_to_char[value] for value in pattern]))
        # generate characters
        for i in range(1000):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(vocab)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        print("\nDone.")
        
print("Hello. As a senior in AP English 4, you are required to post 4 writings stating your opinions Hamlet, posted on edmodo." )
classify = input("Would you like the model to write a post? (Y/N) ")
if classify.lower() == 'y':
    file = input("Input filename (default: edmodowritings " )
    user = writing("aj" ,file)
    user.execute()
else:
    print("Okay. Goodbye")

            


