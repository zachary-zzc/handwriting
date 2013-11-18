from numpy import *
from bpnetwork import bpNetWork
from utils.plot import plot_cost
import struct
import getopt
import sys, os


def load(path, labels=None):
    """Load training set and test set from file
    Input:
        labels: default--all 0-9 numbers will be load
                others, training set and test set of numbers in labels will be load.
                e.g., load((1, 2)) will load training set and test set with labels
                1 and 2
    Output:
        train_X, train_Y: original training set
        test_X, test_Y: original test set
    """
    if labels == None:
        labels = set([i for i in range(10)])
    else:
        labels = set(labels)

    try:
        trainDataFile = open(os.path.join(path,
                                          'train-images.idx3-ubyte'), 'rb')
    except IOError:
        print('Cannot open training img file')
        sys.exit(2)
    trainDataMagicNr, trainDataSize, trainDataRows, trainDataCols \
                                = struct.unpack('>IIII', trainDataFile.read(16))
    try:
        trainLabelFile = open(os.path.join(path,
                                           'train-labels.idx1-ubyte'), 'rb')
    except IOError:
        print('Cannot open training label file')
        sys.exit(2)
    trainLabelMagicNr, trainLabelSize \
                                = struct.unpack('>II', trainLabelFile.read(8))

    train_X = []
    train_Y = []
    while(1):
        bufimg = trainDataFile.read(784)
        buflabel = trainLabelFile.read(1)
        if len(bufimg) == 0 or len(buflabel) == 0:
            break
        if struct.unpack('>1B', buflabel)[0] in labels:
            train_X.append(struct.unpack('>784B', bufimg))
            train_Y.append(struct.unpack('>1B', buflabel))
    trainDataFile.close()
    trainLabelFile.close()

    try:
        testDataFile = open(os.path.join(path,
                                         't10k-images.idx3-ubyte'), 'rb')
    except IOError:
        print('Cannot open test img file')
        sys.exit(2)
    testDataMagicNr, testDataSize, testDataRows, testDataCols \
                                = struct.unpack('>IIII', testDataFile.read(16))
    try:
        testLabelFile = open(os.path.join(path,
                                          't10k-labels.idx1-ubyte'), 'rb')
    except IOError:
        print('Cannot open test img file')
        sys.exit(2)
    testLabelMagicNr, testLabelSize \
                                = struct.unpack('>II', testLabelFile.read(8))
    test_X = []
    test_Y = []
    while(1):
        bufimg = testDataFile.read(784)
        buflabel = testLabelFile.read(1)
        if len(bufimg) == 0 or len(buflabel) == 0:
            break
        if struct.unpack('1B', buflabel)[0] in labels:
            test_X.append(struct.unpack('>784B', bufimg))
            test_Y.append(struct.unpack('1B', buflabel))
    testDataFile.close()
    testLabelFile.close()

    return tuple(train_X), tuple(train_Y), tuple(test_X), tuple(test_Y)



def extractPixelMapFeature(original_x):
    """extract pixel map feature
    Input:
        originalX: original data
    Output:
        X: Pixel map feature extracted from original data
    """
    x = int8(array(original_x) * 4 / 256)
    return tuple(x)

"""Other features can be defined"""


def encodeLabel(original_y, label_num):
    """k-of-1 label coding
    Input:
        original_y: original labels
        label_num: 1-of-K or 1-0
    OutPut:
        y: encoded labels
    """
    y = [0 for i in range(label_num)]
    y[original_y[0]] = 1
    return tuple(y)


def preprocess(originalX, originalY, label_num):
    """convert features and labels to format training input
    Input:
        originalX, originalY: original data
        label_num: classify type number
    Output:
        X: combine all features togather
        Y: encoded label
    """
    X = tuple([extractPixelMapFeature(x) for x in originalX])
    labels = set(originalY)
    if label_num > 2:
        Y = tuple([encodeLabel(y, label_num) for y in originalY])
    elif label_num == 2:
        Y = tuple([(1,) if y == list(labels)[0] else (0,) for y in originalY])
    return X, Y


def initbpNetWork(hidden_layer_num,
                  hidden_unit_list,
                  acfunc_list,
                  output_func,
                  sigma,
                  steps):
    output_func = output_func
    sigma = sigma
    steps = steps
    bp = bpNetWork(hidden_layer_num,
                    hidden_unit_list,
                    acfunc_list,
                    output_func,
                    sigma,
                    steps)
    return bp


def decodeLabel(output_y, label_num):
    """convert bp network output labels to original format
    Input:
        output_y: classification result by bp network
        label_num: 1-of-K label format or two class 0-1 label
    Output:
        y: 1-of-K label format
    """
    if label_num > 2:
        y = [0 for i in range(label_num)]
        index = array(output_y).argmax()
        y[index] = 1
    elif label_num == 2:
        if output_y >= 0.5:
            y = (1, )
        else:
            y = (0, )
    return tuple(y)


def postprocess(output_Y, label_num):
    """everything needs to be done in order to serve measurements
    Input:
        output_Y: classification results by bp network
        label_num: determing 1-of-K decoding or one classifici
    Output:
        Y: 1-of-K label format
    """
    Y = tuple([decodeLabel(y, label_num) for y in output_Y])
    return Y



def measureCorrectRate(test_Y, output_Y):
    """meature the correct rate
    Input:
        test_Y: test data labels
        output_Y: classification result by bp network
    Output:
        correctRate: correct rate of bp network
    """
    # raise error if test set and output set don't have same length
    size = len(test_Y)
    eqlist = [test_Y[i] == output_Y[i] for i in range(size)]
    count = len(nonzero(array(eqlist) == True)[0])
    return 100 * count / size


#def measurements():
    """all kinds of measurements
    Input:
        bp: bp network instance
        test_Y: test data labels
        Y: classification result by bp network
    Output:
        performance of bp network
    """
#    pass


def main(argv=None):
    shortopt = 'hu:f:s:t:'
    longopt = ['help', 'unit', 'function', 'sigma', 'steps']
    if argv == None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], shortopt, longopt)
    except getopt.GetoptError:
        print('for help use --help')
        sys.exit(2)
    # init bp network param
    hidden_layer_num = 1
    hidden_unit_list = (128, )
    acfunc_list = ('tanh', )
    output_func = 'sigmoid'
    sigma = 1e-4
    steps = 1000
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('help!')
            sys.exit()
        elif opt in ('-u', '--unit'):
            hidden_unit_list = (int(arg), )
        elif opt in ('-f', '--function'):
            output_func = arg
        elif opt in ('-s', '--sigma'):
            sigma = arg
        elif opt in ('-t', '--steps'):
            steps = arg
    print('#####Loading Data.#####')
    label_set = set([i for i in range(10)])
    #label_set = set([0, 1, 2])
    label_type = len(label_set)
    train_X, train_Y, test_X, test_Y = load('./data/', label_set)

    print('#####Training Data Preprocess.#####')
    train_X, train_Y = preprocess(train_X[0: 10000], train_Y[0: 10000], label_type)
    print('Training Data Length: {}\n\
Test Data Length: {}\n'.format(len(train_Y),
                               len(test_Y)))

    print('#####Init bpNetwork.#####')
    print('NetWork Parameter:\n\
Hidden Layer Number: {}\n\
Hidden Layer Unit List: {}\n\
Hidden Layer Active Function List: {}\n\
Output Active Function: {}\n\
Gradient Descent Learning Rate: {}\n\
Iterator Steps: {}\n'.format(hidden_layer_num,
                             hidden_unit_list,
                             acfunc_list,
                             output_func,
                             sigma,
                             steps))
    bp = initbpNetWork(hidden_layer_num,
                       hidden_unit_list,
                       acfunc_list,
                       output_func,
                       sigma,
                       steps)

    print('#####Start Training.#####')
    bp.train(train_X, train_Y)
    plot_cost(bp.debug_x, bp.debug_y)
    print('#####End Training.#####')

    print('#####Start Test.#####')

    print('#####Preprocess Test Image.#####')
    test_X, test_Y = preprocess(test_X, test_Y, label_type)
    print('#####Start Simulate.#####')
    output_Y = bp.simulate(test_X)
    print('#####Check Correct Rate.#####')
    label_num = len(set(test_Y))
    output_Y = postprocess(output_Y, label_num)
    correctRate = measureCorrectRate(output_Y, test_Y)
    print('Correct Rate of Classifier: {:.2f}'.format(correctRate))
    del bp


if __name__ == '__main__':
    main()

