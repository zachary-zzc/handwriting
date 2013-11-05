from numpy import *
from tools import get_acfunc, get_gradfunc
import matplotlib.pyplot as plt

class layer:
    def __init__(self, sigma, unit_num, acfunc):
        """parameters in Layer:
            w: coefficients to transform activated input to output, this parameter in output layer will always be []
            sigma: learning rate
            unit_num: unit number regardless bias
            acfunc: active function
            gradfunc: gradiant function of active function
            X: input to this layer, without bias
            T: output
        """
        self.sigma = sigma
        self.unit_num = unit_num
        self.acfunc = get_acfunc(acfunc)
        self.gradfunc = get_gradfunc(acfunc)
        self.w = []
        self.X = []
        self.T = []


    def setw(self, w):
        self.w = w


    def forward(self, X):
        self.X = X
        Z = self.acfunc(self.X)
        return dot(c_[ones((len(Z), 1)), Z], self.w)


    def backward(self, delta):
        # next layer's delta will be used to calculate this layer's delta
        return self.gradfunc(self.X) * dot(delta, self.w[1:, :].T)


    def update(self, delta):
        # next layer's delta will be used to update this layer's coefficients
        self.w = self.w + \
                 self.sigma * \
                 dot(self.acfunc(c_[ones((len(self.X), 1)), self.X]).T, delta)


    def Jacobian():
        pass


    def Hession():
        pass


class bpNetWork:
    def __init__(self,
                 hidden_layer_num,
                 hidden_unit_num_list,
                 acfunc_list,
                 output_func,
                 sigma,
                 steps):
        """paremeters in bpNetWork:
            hidden_layer_num: hidden layer number
            hidden_layers: list of hidden layer
            delta_list: list of errors in each layer
            input_layer: input layer
            output_func: output layer function
            sigma: learning rate
            steps: largest iterator steps
            Y: output calculated by forward propagatin
            cost: last cost value after training
            debug_X: iterate time
            debug_y: cost function after each iterate
        """
        if hidden_layer_num > len(acfunc_list):
            for iter in range(len(acfunc_list), hidden_layer_num):
                acfunc_list.append('tanh')

        self.hidden_layer_num = hidden_layer_num
        self.hidden_layers = tuple([layer(sigma,
                                   hidden_unit_num_list[i],
                                   acfunc_list[i])
                                   for i in range(hidden_layer_num)])
        self.output_func = get_acfunc(output_func)
        self.sigma = sigma
        self.steps = steps
        self.delta_list = []
        self.debug_x = []
        self.debug_y = []


    def initw(self, X, T):
        self.input_layer = layer(self.sigma, len(X[0])+1, 'line')
        # initial input layer's coefficients
        w = random.rand(len(X[0])+1, self.hidden_layers[0].unit_num) * 2 - 1
        self.input_layer.setw(w)
        # initial hidden layers' coefficients
        for iter in range(self.hidden_layer_num - 1):
            w = random.rand(self.hidden_layers[iter].unit_num + 1,
                            self.hidden_layers[iter+1].unit_num) * 2 - 1
            self.hidden_layers[iter].setw(w)
        # initial last hidden layer's coefficients
        w = random.rand(self.hidden_layers[self.hidden_layer_num-1].unit_num + 1,
                        len(T[0])) * 2 - 1
        self.hidden_layers[self.hidden_layer_num-1].setw(w)


    def forward_propagation(self, X):
        # calculate input for first layer
        a = self.input_layer.forward(X)
        # forward propagate through hidden layers
        for iter in range(self.hidden_layer_num):
            a = self.hidden_layers[iter].forward(a)
        # calculate output
        # self.Y = self.output_func(self.hidden_layers[self.hidden_layer_num-1].forward(a))
        self.Y = self.output_func(a)


    def backward_propagation(self, T):
        # output layer's delta
        self.delta_list.insert(0, T - self.Y)
        # backward propagate through hidden layers
        for iter in range(self.hidden_layer_num-1, -1, -1):
            delta = self.hidden_layers[iter].backward(self.delta_list[0])
            self.delta_list.insert(0, delta)


    def updatew(self):
        self.input_layer.update(self.delta_list[0])
        for iter in range(self.hidden_layer_num):
            self.hidden_layers[iter].update(self.delta_list[iter+1])


    def costFunc(self, T):
        return ((T - self.Y) ** 2).sum() / 2



    def gradcheck(self):
        pass


    def train(self, X, T):
        # init coefficients
        X = array(X)
        T = array(T)
        self.initw(X, T)
        # clear debug_x, debug_y
        self.debug_x = []
        self.debug_y = []
        for step in range(self.steps):
            self.forward_propagation(X)
            self.backward_propagation(T)
            # update coefficients for each layer
            self.updatew()
            # calculate cost function
            self.debug_x.append(step)
            self.debug_y.append(self.costFunc(T))
        # record the last cost value
        self.cost = self.costFunc(T)


    def simulate(self, X):
        self.forward_propagation(X)
        return self.Y


    def test_plot(self, X, T):
        steps = linspace(-1, 1, 100)
        self.simulate(steps)
        plt.figure()
        plt.subplot(111)
        plt.scatter(X, T)
        plt.plot(steps, self.Y)
        plt.show()


    def plot_cost(self):
        plt.figure()
        plt.subplot(111)
        plt.ylim(0, 2)
        plt.plot(self.debug_x, self.debug_y)
        plt.show()




