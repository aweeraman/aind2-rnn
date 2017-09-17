import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
import keras
import re


# Fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(np.array(series[i:i+window_size]))

    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y


# Build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### Return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    text = re.sub(r'[^a-zA-Z\-!,\.\:\;\?\s]+', '', text)
    text = re.sub(r'--', '', text)
    text = re.sub(r'\d\.\d', '', text)   # examples like 1.e.2
    text = re.sub(r'\.e\.\.', '', text)
    text = re.sub(r'iii\.', '', text)
    text = re.sub(r'businesspglaf\.org\.', '', text)

    return text


### Fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text) - window_size, step_size):
        inputs.append(np.array(text[i:i+window_size]))
        outputs.append(text[i+window_size])

    # reshape each 
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)

    return inputs,outputs

# Build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
