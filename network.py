import numpy as np
import skimage.measure
import pickle
from layers import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]

    def train_save(self, training_data, epochs, mini_batch_size, lr, validation_data, test_data, test=True, save=0):
        # training_x shape is [num, width, height, chanel], training_y shape is [num, label],same as validation & test
        # save=0:no save;save=1:save after training;save=2:save the best result
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        num_training_batches = int(training_y.shape[0] / mini_batch_size)
        num_validation_batches = int(validation_y.shape[0] / mini_batch_size)
        num_test_batches = int(test_y.shape[0] / mini_batch_size)

        best_accuracy, best_wide_accuracy = [0, 0]
        rms = RMSProp(lr=0.001, beta=0.9, epsilon=1e-8)
        for epoch in range(epochs):
            # training
            epoch_loss = 0
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 20 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                loss_value = self.train_once(training_x, training_y, lr, minibatch_index, None)
                epoch_loss = epoch_loss + loss_value / num_training_batches
            print("epoch", epoch+1)
            print("loss", epoch_loss)
            # accuracy
            temp1, temp2 = [[], []]
            for j in range(num_validation_batches):
                temp1_1, temp2_1 = self.accuracy(validation_x, validation_y, j)
                temp1.append(temp1_1)
                temp2.append(temp2_1)
                validation_accuracy, wide_accuracy = [np.mean(temp1), np.mean(temp2)]
            # save
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                if save == 2:
                    file = open('./best_acc_params3.pickle', 'wb')
                    pickle.dump(self.params, file)
                    file.close()
            if wide_accuracy > best_wide_accuracy:
                best_wide_accuracy = wide_accuracy
                if save == 2:
                    file = open('./best_wide_params3.pickle', 'wb')
                    pickle.dump(self.params, file)
                    file.close()

            print("Epoch {0}: validation accuracy {1:.2%}, wide accuracy {2:.2%}".format(epoch+1, validation_accuracy, wide_accuracy))

            #validation_accuracy = np.mean([self.accuracy(validation_x, validation_y, j) for j in range(num_validation_batches)])
            #print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))

        print(" Best validation accuracy {0:.2%}, Best wide accuracy {1:.2%}".format(best_accuracy, best_wide_accuracy))
        print("Finished training network.")
        if save == 1:
            file = open('./params.pickle', 'wb')
            pickle.dump(self.params, file)
            file.close()

        if test == True:
            test_accuracy = np.mean([self.accuracy(validation_x, validation_y, j) for j in range(num_test_batches)])
            print("Final test accuracy is {0:.2%}".format(test_accuracy))

    def load_save(self, training_data, epochs, mini_batch_size, lr, validation_data, test_data, test=True, save=0, path='./params.pickle'):
        # training_x shape is [num, width, height, chanel], training_y shape is [num, label],same as validation & test
        # save=0:no save;save=1:save after training;save=2:save the best result
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        num_training_batches = int(training_y.shape[0] / mini_batch_size)
        num_validation_batches = int(validation_y.shape[0] / mini_batch_size)
        num_test_batches = int(test_y.shape[0] / mini_batch_size)
        # load parameters
        with open(path, 'rb') as file:
            loaded_params = pickle.load(file)
        self.params = loaded_params
        for i in range(len(self.layers)):
            self.layers[i].w = self.params[2 * i]
            self.layers[i].b = self.params[2 * i + 1]

        best_accuracy, best_wide_accuracy = [0, 0]
        rms = RMSProp(lr=0.001, beta=0.9, epsilon=1e-8)
        for epoch in range(epochs):
            # training
            epoch_loss = 0
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 20 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                loss_value = self.train_once(training_x, training_y, lr, minibatch_index, None)
                epoch_loss = epoch_loss + loss_value / num_training_batches
            print("epoch", epoch + 1)
            print("loss", epoch_loss)
            # accuracy
            temp1, temp2 = [[], []]
            for j in range(num_validation_batches):
                temp1_1, temp2_1 = self.accuracy(validation_x, validation_y, j)
                temp1.append(temp1_1)
                temp2.append(temp2_1)
                validation_accuracy, wide_accuracy = [np.mean(temp1), np.mean(temp2)]
            # save
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                if save == 2:
                    file = open('./best_acc_params3_1.pickle', 'wb')
                    pickle.dump(self.params, file)
                    file.close()
            if wide_accuracy > best_wide_accuracy:
                best_wide_accuracy = wide_accuracy
                if save == 2:
                    file = open('./best_wide_params3_1.pickle', 'wb')
                    pickle.dump(self.params, file)
                    file.close()

            print("Epoch {0}: validation accuracy {1:.2%}, wide accuracy {2:.2%}".format(epoch+1, validation_accuracy,
                                                                                         wide_accuracy))

            # validation_accuracy = np.mean([self.accuracy(validation_x, validation_y, j) for j in range(num_validation_batches)])
            # print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))

        print(" Best validation accuracy {0:.2%}, Best wide accuracy {1:.2%}".format(best_accuracy, best_wide_accuracy))
        print("Finished training network.")
        if save == 1:
            file = open('./params.pickle', 'wb')
            pickle.dump(self.params, file)
            file.close()

        if test == True:
            test_accuracy = np.mean([self.accuracy(test_x, test_y, j) for j in range(num_test_batches)])
            print("Final test accuracy is {0:.2%}".format(test_accuracy))

    def load_test(self, mini_batch_size, test_data, path='./params.pickle'):
        # training_x shape is [num, width, height, chanel], training_y shape is [num, label],same as validation & test
        test_x, test_y = test_data
        num_test_batches = int(test_y.shape[0] / mini_batch_size)
        # load parameters
        with open(path, 'rb') as file:
            loaded_params = pickle.load(file)
        self.params = loaded_params
        for i in range(len(self.layers)):
            self.layers[i].w = self.params[2 * i]
            self.layers[i].b = self.params[2 * i + 1]
        # accuracy
        #temp1 = []
        temp2 = []
        #confidence = []
        for j in range(num_test_batches):
            #temp1_1, temp2_1, confj, predj = self.test_accuracy(test_x, test_y, j)
            temp2_1, predj1, confj1, predj2, confj2, predj3, confj3 = self.test_accuracy(test_x, test_y, j)
            #temp1.append(temp1_1)
            temp2.append(temp2_1)
            for ba in range(mini_batch_size):
                print("num", j*mini_batch_size+ba+1)
                print("prediction1:", predj1[ba],"confidence1:", confj1[ba])
                print("prediction2:", predj2[ba],"confidence2:", confj2[ba])
                print("prediction3:", predj3[ba], "confidence3:", confj3[ba])
                print("true label:", np.argmax(test_y[j * mini_batch_size + ba]) + 1)
            #confidence = confidence + confj
            #test_accuracy, wide_accuracy = [np.mean(temp1), np.mean(temp2)]
            wide_accuracy = np.mean(temp2)
        #print("Final test accuracy is {0:.2%}".format(test_accuracy))
        print("Final wide accuracy is {0:.2%}".format(wide_accuracy))

    def train_once(self, input, label, lr, i, rms):
        # calculate the loss, update the parameters for once
        inpt = input[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
        y = label[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
        loss_value = self.Forward(inpt, y)[1]
        grads = self.grad(y)
        #self.params = rms.get_update(self.params, grads)
        for num in range(len(self.params)):
            self.params[num] = self.params[num] - lr * grads[num]
        for num in range(len(self.layers)):
            self.layers[num].w = self.params[2*num]
            self.layers[num].b = self.params[2*num+1]
        return loss_value

    def accuracy(self, input, label, i):
        # calculate the output, and get the accuracy
        inpt = input[i * self.mini_batch_size:(i + 1) * self.mini_batch_size]
        y = label[i * self.mini_batch_size:(i + 1) * self.mini_batch_size]
        self.Forward(inpt, y)
        accuracy = self.layers[-1].accuracy(y)
        wide_accuracy = self.layers[-1].wide_accuracy(y)
        return [accuracy, wide_accuracy]

    def test_accuracy(self, input, label, i):
        # calculate the output, and get the accuracy
        inpt = input[i * self.mini_batch_size:(i + 1) * self.mini_batch_size]
        y = label[i * self.mini_batch_size:(i + 1) * self.mini_batch_size]
        self.Forward(inpt, y)
        #prediction = np.argmax(output, axis=1)
        #confidence = [output[num][prediction[num]] for num in range(len(output))]
        accuracy = self.layers[-1].test_accuracy(y)
        wide_accuracy, prediction1, confidence1, prediction2, confidence2, prediction3, confidence3 = self.layers[-1].test_wide_accuracy(y)
        return [wide_accuracy, prediction1, confidence1, prediction2, confidence2, prediction3, confidence3]

    def Forward(self, inpt, y):
        # calculate the loss, and output
        init_layer = self.layers[0]
        init_layer.forward(inpt, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.forward(prev_layer.output, self.mini_batch_size)
        self.output = self.layers[-1].output
        loss_value = self.layers[-1].loss(y)
        return [self.output, loss_value]

    def grad(self, y):
        # calculate the grad
        rev_grads = []
        grads = []
        last_layer = self.layers[-1]
        grad_param, grad_y = last_layer.back_prop(y)
        rev_grads.append(grad_param[1])
        rev_grads.append(grad_param[0])
        for j in range(len(self.layers) - 2, -1, -1):
            grad_param, grad_y = self.layers[j].back_prop(grad_y)
            rev_grads.append(grad_param[1])
            rev_grads.append(grad_param[0])
        for j in range(len(rev_grads)-1, -1, -1):
            grads.append(rev_grads[j])
        return grads


class Adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def get_update(self, params, grads):
        original_shapes = [x.shape for x in params]
        lr = self.lr
        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t))/(1. - np.power(self.beta_1, t)))
        self.ms = [np.zeros(p.shape) for p in params]
        self.vs = [np.zeros(p.shape) for p in params]
        ret = [None] * len(params)

        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t
        self.iterations += 1
        return ret

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.iterations = 0
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

    def get_update(self, params, grads):
        original_shapes = [x.shape for x in params]
        t = self.iterations + 1
        self.ms = [np.zeros(p.shape) for p in params]
        ret = [None] * len(params)

        for i, p, g, m in zip(range(len(params)), params, grads, self.ms):
            m_t = (self.beta * m) + (1. - self.beta) * np.square(g)
            p_t = p - self.lr * g / (np.sqrt(m_t) + self.epsilon)
            self.ms[i] = m_t
            ret[i] = p_t
        self.iterations += 1
        return ret


