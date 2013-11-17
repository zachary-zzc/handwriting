from numpy import tanh, exp

def sigmoid(a):
    return 1/(1 + exp(-a))

def line(a):
    return a

def gradtanh(a):
    return 1 - tanh(a) ** 2

def gradsigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

def gradline(a):
    return 1

def step_func(a):
    if a >= 0:
        return 1
    return 0
