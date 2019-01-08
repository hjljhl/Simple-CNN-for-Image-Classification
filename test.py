import numpy as np
import skimage.measure
import pickle
from readlabel import read_image
from network import Network
from layers import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU, Sigmoid

whole_data = read_image(path1 = 'test_images/', path2 = './test_annotation', data_size = 1050)

whole_x = whole_data[0]
mean = whole_x.mean(axis=0)
std = whole_x.std(axis=0)
whole_x = (whole_x - mean) / std
whole_y = whole_data[1]
test_x = whole_x
test_y = whole_y
test_data = [test_x, test_y]

mini_batch_size = 1

# final
net = Network([ConvPoolLayer(filter_shape=(5, 5, 3, 9), image_shape=(mini_batch_size, 64, 64, 3), poolsize=2, activation_fn=ReLU),
               ConvPoolLayer(filter_shape=(5, 5, 9, 18), image_shape=(mini_batch_size, 30, 30, 9), poolsize=2, activation_fn=ReLU),
               ConvPoolLayer(filter_shape=(4, 4, 18, 36), image_shape=(mini_batch_size, 13, 13, 18), poolsize=2, activation_fn=ReLU),
               FullyConnectedLayer(n_in=900, n_out=225, activation_fn=ReLU),
               FullyConnectedLayer(n_in=225, n_out=50, activation_fn=ReLU),
               SoftmaxLayer(n_in=50, n_out=20, activation_fn=None)], mini_batch_size)

print('start')
net.load_test(mini_batch_size, test_data, path='./finalparams_noact.pickle')

