import numpy as np
import skimage.measure
import pickle
from network import Network
from layers import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU, Sigmoid

with open('./whole_data.pickle', 'rb') as file:
    whole_data = pickle.load(file)
    
whole_x = whole_data[0]
mean = whole_x.mean(axis=0)
std = whole_x.std(axis=0)
whole_x = (whole_x - mean) / std
whole_y = whole_data[1]
training_x = whole_x[:9000]
training_y = whole_y[:9000]
training_data = [training_x, training_y]
validation_x = whole_x[9000:]
validation_y = whole_y[9000:]
validation_data = [validation_x, validation_y]
test_x = whole_x[9000:]
test_y = whole_y[9000:]
test_data = [test_x, test_y]

# final
net = Network([ConvPoolLayer(filter_shape=(5, 5, 3, 9), image_shape=(mini_batch_size, 64, 64, 3), poolsize=2, activation_fn=ReLU),
               ConvPoolLayer(filter_shape=(5, 5, 9, 18), image_shape=(mini_batch_size, 30, 30, 9), poolsize=2, activation_fn=ReLU),
               ConvPoolLayer(filter_shape=(4, 4, 18, 36), image_shape=(mini_batch_size, 13, 13, 18), poolsize=2, activation_fn=ReLU),
               FullyConnectedLayer(n_in=900, n_out=225, activation_fn=ReLU),
               FullyConnectedLayer(n_in=225, n_out=50, activation_fn=ReLU),
               SoftmaxLayer(n_in=50, n_out=20, activation_fn=None)], mini_batch_size)

print('start')
net.train_save(training_data, 13, mini_batch_size, 0.001, validation_data, test_data, test=False, save=2)

