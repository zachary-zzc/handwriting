from numpy import tanh, exp

def get_acfunc(acfunc):
    if acfunc == 'tanh':
        return tanh
    if acfunc == 'sigmoid':
        return sigmoid
    if acfunc == 'line':
        return line

def get_gradfunc(acfunc):
    if acfunc == 'tanh':
        return gradtanh
    elif acfunc == 'sigmoid':
        return gradsigmoid
    elif acfunc == 'line':
        return gradline

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
