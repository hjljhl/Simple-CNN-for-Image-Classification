import numpy as np
import skimage.measure
import pickle

def ReLU(z):
    k1 = z>0
    k2 = 1-k1
    z2 = z * k1 + z * k2 / 100
    return z2
def Sigmoid(z):
    z1 = np.float64(z)
    if z1.all()>=0:
        return 1/(1+np.exp(-z1))
    else:
        return np.exp(z1)/(1+np.exp(z1))
def Softmax(z):
    z1 = np.zeros(z.shape)
    z2 = np.zeros(z.shape)
    for i in range(z.shape[0]):
        zi = z[i]
        z1[i] = np.exp(zi) / np.sum(np.exp(zi))
        z2[i] = zi / np.sum(z)
    return z1
def ReLU_back(y):
    k1 = y > 0
    k2 = 1 - k1
    y2 = np.ones(y.shape)*k1 + np.ones(y.shape)*k2/100
    return y2
def Sigmoid_back(y): return y*(1-y)

class ConvPoolLayer(object):
    '''filter_shape=[width, height, in_chanel, out_chanel];image_shape=[minibatch,width ,height,in_chanel]'''
    def __init__(self, filter_shape, image_shape, poolsize=1, activation_fn=ReLU):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn =activation_fn
        # initialize weights and biases
        n_out = filter_shape[0] * filter_shape[1] * filter_shape[2]
        self.w = np.array(np.random.normal(loc=0, scale=np.sqrt(2.0 / n_out), size=filter_shape), dtype=float)
        self.b = np.array(np.random.normal(loc=0, scale=1.0, size=(filter_shape[3],)), dtype=float)
        self.params = [self.w, self.b]

    def forward(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = self.conv2d(x=self.inpt, stride=1)
        self.pool_in = self.activation_fn(conv_out)
        self.output = self.pool2d(self.pool_in, self.poolsize)

    def conv2d(self, x, stride):
        self.stride = stride
        self.col_weights = self.w.reshape([-1, self.filter_shape[3]])
        self.eta = np.zeros((self.image_shape[0], int((self.image_shape[1] - self.filter_shape[0]) / self.stride + 1),
                             int((self.image_shape[2] - self.filter_shape[1]) / self.stride + 1), self.filter_shape[3]))
        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.image_shape[0]):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.filter_shape[1], self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, self.col_weights) + self.b, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def pool2d(self, inpt, p):
        pool_out = np.zeros([inpt.shape[0], int(inpt.shape[1] / p), int(inpt.shape[2] / p), inpt.shape[3]])
        self.max_index = np.zeros([inpt.shape[0], int(inpt.shape[1] / p), int(inpt.shape[2] / p), inpt.shape[3]])
        for i in range(inpt.shape[0]):
            for j in range(inpt.shape[3]):
                for k in range(int(inpt.shape[1] / p)):
                    for l in range(int(inpt.shape[2] / p)):
                        temp = inpt[i, p*k:(p*k+p), p*l:(p*l+p), j]
                        pool_out[i, k, l, j] = np.max(temp)
                        self.max_index[i, k, l, j] = np.argmax(temp)
        return pool_out

    def pool_back(self, y, p):
        yt = np.zeros([y.shape[0], y.shape[1]*p, y.shape[2]*p, y.shape[3]])
        for i in range(y.shape[0]):
            for j in range(y.shape[3]):
                for k in range(y.shape[1]):
                    for l in range(y.shape[2]):
                        [x_i, y_i] = [self.max_index[i, k, l, j]//p, self.max_index[i, k, l, j]%p]
                        yt[i, int(p*k+x_i), int(p*l+y_i), j] = y[i, k, l, j]
        return yt

    def back_prop(self, L_y):
        L_y = L_y.reshape(self.output.shape)
        L_yt = self.pool_back(L_y, self.poolsize)
        if self.activation_fn == ReLU:
            yt_zt = ReLU_back(self.pool_in)
        else:
            yt_zt = Sigmoid_back(self.pool_in)
        L_z = (L_yt*yt_zt).reshape((self.image_shape[0], -1, self.filter_shape[3]))
        L_col_w = np.zeros(self.col_weights.shape)
        for i in range(self.image_shape[0]):
            L_col_w = L_col_w+np.dot(self.col_image[i].T, L_z[i])
        L_w = L_col_w.reshape(self.filter_shape)
        L_b = np.sum(L_z, axis=(0, 1))/self.image_shape[0]
        grad_param = [L_w, L_b]
        L_col_y = np.zeros(self.col_image.shape)
        for i in range(self.image_shape[0]):
            L_col_y[i] = L_col_y[i]+np.dot(L_z[i], self.col_weights.T)
        grad_y = in_im2col(L_col_y, self.filter_shape[1], self.filter_shape[1], self.image_shape)
        return [grad_param, grad_y]


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=ReLU):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = np.array(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)), dtype=float)
        self.b = np.array(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=float)
        self.params = [self.w, self.b]

    def forward(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.inpt_shape = self.inpt.shape
        self.output = self.activation_fn(np.dot(self.inpt, self.w) + self.b)
        self.y_out = np.argmax(self.output, axis=1)

    def back_prop(self, L_y):
        if self.activation_fn == ReLU:
            y_z = ReLU_back(self.output)
        else:
            y_z = Sigmoid_back(self.output)
        L_z = L_y*y_z
        L_w = np.dot(self.inpt.T, L_z)
        grad_y = np.dot(L_z, self.w.T)
        L_b = np.sum(L_z, axis=0)/L_z.shape[0]
        grad_param = [L_w, L_b]
        return [grad_param, grad_y]


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, activation_fn=ReLU):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = np.array(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / self.n_in), size=(n_in, n_out)), dtype=float)
        self.b = np.array(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=float)
        self.params = [self.w, self.b]

    def forward(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        if self.activation_fn == ReLU or self.activation_fn == Sigmoid:
            self.y_1 = self.activation_fn(np.dot(self.inpt, self.w) + self.b)
        self.y_1 = np.dot(self.inpt, self.w) + self.b
        self.output = Softmax(self.y_1)
        self.y_out = np.argmax(self.output, axis=1)

    def back_prop(self, y):
        L_y = self.output - y
        if self.activation_fn == ReLU:
            y_z = ReLU_back(self.y_1)
        if self.activation_fn == Sigmoid:
            y_z = Sigmoid_back(self.y_1)
        else:
            y_z = 1
        L_z = L_y*y_z
        L_w = np.dot(self.inpt.T, L_z)
        L_b = np.sum(L_z, axis=0)/L_z.shape[0]
        grad_param = [L_w, L_b]
        grad_y = np.dot(L_z, self.w.T)
        return [grad_param, grad_y]

    def loss(self, y):
        # y is a matrix, (minibatchsize, label[])
        return - np.sum(np.log(self.output) * y)

    def accuracy(self, y):
        label = np.argmax(y, axis=1)
        return np.mean(np.equal(label, self.y_out))

    def test_accuracy(self, y):
        max = np.max(self.output, axis=1)
        # if confidence < 0.05 prediction is UNKNOWN
        self.final_out = ((max - 0.05) > 0) * self.y_out + ((max - 0.05) <= 0) * 20
        label = np.argmax(y, axis=1)
        label = (np.sum(y, axis=1)==0)*20+label
        return np.mean(np.equal(label, self.final_out))

    def wide_accuracy(self, y):
        label = np.argmax(y, axis=1)
        acc = np.zeros(label.shape)
        for i in range(len(self.output)):
            output = self.output[i]
            max_index = np.argmax(output)
            output[max_index] = output[max_index] - 1
            second_index = np.argmax(output)
            output[second_index] = output[second_index]-1
            third_index = np.argmax(output)
            if (label[i] == max_index)|(label[i] == second_index)|(label[i] == third_index):
                acc[i] = 1
        return np.mean(acc)

    def test_wide_accuracy(self, y):
        label = np.argmax(y, axis=1)
        label = (np.sum(y, axis=1) == 0) * 20 + label
        acc = np.zeros(label.shape)
        firstone = []
        secondone = []
        thirdone = []
        firstconf = []
        secondconf = []
        thirdconf = []
        for i in range(len(self.output)):
            output = self.output[i]
            max_index = np.argmax(output)
            first_max = np.max(output)
            output[max_index] = output[max_index] - 1
            second_index = np.argmax(output)
            second_max = np.max(output)
            output[second_index] = output[second_index]-1
            third_index = np.argmax(output)
            third_max = np.max(output)
            # if confidence < 0.05 prediction is UNKNOWN
            if first_max<0.05:
                firstone.append(21)
                secondone.append(max_index+1)
                thirdone.append(second_index+1)
                firstconf.append("-")
                secondconf.append(first_max)
                thirdconf.append(second_max)
            elif second_max<0.05:
                firstone.append(max_index + 1)
                secondone.append(21)
                thirdone.append(second_index + 1)
                firstconf.append(first_max)
                secondconf.append("-")
                thirdconf.append(second_max)
            elif third_max<0.05:
                firstone.append(max_index + 1)
                secondone.append(second_index + 1)
                thirdone.append(21)
                firstconf.append(first_max)
                secondconf.append(second_max)
                thirdconf.append("-")
            else:
                firstone.append(max_index + 1)
                secondone.append(second_index + 1)
                thirdone.append(third_index+1)
                firstconf.append(first_max)
                secondconf.append(second_max)
                thirdconf.append(third_max)
            if (label[i] == firstone[i]-1) | (label[i] == secondone[i]-1) | (label[i] == thirdone[i]-1):
                    acc[i] = 1
        return [np.mean(acc), firstone, firstconf, secondone, secondconf, thirdone, thirdconf]

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col

def in_im2col(image_col, ksize, stride, image_shape):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image = np.zeros(image_shape)
    for n in range(image_col.shape[0]):
        for i in range(0, image_shape[1] - ksize + 1, stride):
            for j in range(0, image_shape[2] - ksize + 1, stride):
                t = int(i * (image_shape[2] - ksize + 1) / (2 * stride) + j / stride)
                temp = image_col[n, t].reshape((1, ksize, ksize, image_shape[3]))
                image[n, i:i + ksize, j:j + ksize, :] = temp
    return image


