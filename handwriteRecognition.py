from numpy import *
from bpnetwork import bpNetWork
import struct
import getopt
import sys


def load():
    """Load training set and test set from file
    Input:
        None
    Output:
        train_X, train_Y: original training set
        test_X, test_Y: original test set
    """
    trainDataFile = open('train-images.idx3-ubyte', 'rb')
    trainDataMagicNr, trainDataSize, trainDataRows, trainDataCols \
                                = struct.unpack('>IIII', trainDataFile.read(16))
    train_X = []
    while(1):
        buf = trainDataFile.read(784)
        if len(buf) < 784:
            break
        train_X.append(struct.unpack('>784B', buf))
    trainDataFile.close()

    trainLabelFile = open('train-labels.idx1-ubyte', 'rb')
    trainLabelMagicNr, trainLabelSize \
                                = struct.unpack('>II', trainLabelFile.read(8))
    train_Y = []
    while(1):
        buf = trainLabelFile.read(1)
        if len(buf) == 0:
            break
        train_Y.append(struct.unpack('>1B', buf))
    trainLabelFile.close()

    testDataFile = open('t10k-images.idx3-ubyte', 'rb')
    testDataMagicNr, testDataSize, testDataRows, testDataCols \
                                = struct.unpack('>IIII', testDataFile.read(16))
    test_X = []
    while(1):
        buf = testDataFile.read(784)
        if len(buf) < 784:
            break
        test_X.append(struct.unpack('>784B', buf))
    testDataFile.close()

    testLabelFile = open('t10k-labels.idx1-ubyte', 'rb')
    testLabelMagicNr, testLabelSize \
                                = struct.unpack('>II', testLabelFile.read(8))
    test_Y = []
    while(1):
        buf = testLabelFile.read(1)
        if len(buf) == 0:
            break
        test_Y.append(struct.unpack('1B', buf))
    testLabelFile.close()

    return tuple(train_X), tuple(train_Y), tuple(test_X), tuple(test_Y)



def extractPixelMapFeature(original_x):
    """extract pixel map feature
    Input:
        originalX: original data
    Output:
        X: Pixel map feature extracted from original data
    """
    x = []
    for pixel in original_x:
        if 0 <= pixel < 255/4:
            x.append(0)
        elif 255/4 <= pixel < 255/2:
            x.append(1)
        elif 255/2 <= pixel < 255*3/4:
            x.append(2)
        elif 255*3/4 <= pixel <= 255:
            x.append(3)
    return tuple(x)

"""Other features can be defined"""


def encodeLabel(original_y):
    """k-of-1 label coding
    Input:
        original_y: original labels
    OutPut:
        y: encoded labels
    """
    y = [0 for i in range(10)]
    y[original_y[0]] = 1
    return tuple(y)


def preprocess(originalX, originalY):
    """convert features and labels to format training input
    Input:
        originalX, originalY: original data
    Output:
        X: combine all features togather
        Y: encoded label
    """
    X = tuple([extractPixelMapFeature(x) for x in originalX])
    Y = tuple([encodeLabel(y) for y in originalY])
    return X, Y


def initbpNetWork(hidden_layer_num,
                  hidden_unit_list,
                  acfunc_list,
                  output_func=None,
                  sigma=None,
                  steps=None):
    output_func = output_func or 'sigmoid'
    sigma = sigma or 1e-2
    steps = steps or 10000
    bp = bpNetWork(hidden_layer_num,
                    hidden_unit_list,
                    acfunc_list,
                    output_func,
                    sigma,
                    steps)
    return bp


def decodeLabel(output_y):
    """convert bp network output labels to original format
    Input:
        output_y: classification result by bp network
    Output:
        y: 1-of-K label format
    """
    y = (array(output_y).argmax(),)
    return y


def postprocess(output_Y):
    """everything needs to be done in order to serve measurements
    Input:
        output_Y: classification results by bp network
    Output:
        Y: 1-of-K label format
    """
    Y = tuple([decodeLabel(y) for y in output_Y])
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
    count = 0
    for iter in range(size):
        if test_Y[iter] == output_Y[iter]:
            count += 1
    return count/size


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
    if argv == None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], 'hn', ['help'])
    except getopt.GetoptError:
        print('for help use --help')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('help!')
            sys.exit()
        elif opt == '-n':
            hidden_layer_number = arg

    print('#####Loading Data.#####')
    origtrain_X, origtrain_Y, origtest_X, origtest_Y = load()
    print('Training Data Length: {}\n \
            Test Data Length: {}\n'.format(len(origtrain_Y),
                                           len(origtest_Y)))

    print('#####Training Data Preprocess.#####')
    train_X, train_Y = preprocess(origtrain_X, origtrain_Y)

    print('#####Init bpNetwork.#####')
    hidden_layer_num = 1
    hidden_unit_list = (128, )
    acfunc_list = ('tanh', )
    output_func = 'sigmoid'
    sigma = 1e-2
    steps = 10000
    print('NetWork Parameter:\n \
           Hidden Layer Number: {}\n \
           Hidden Layer Unit List: {}\n \
           Hidden Layer Active Function List: {}\n \
           Output Active Function: {}\n \
           Gradient Descent Learning Rate: {}\n \
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
    bp.train(train_X[0:100], train_Y[0:100])

    print('#####End Training.#####')

    print('#####Start Test.#####')
    print('#####Preprocess Test Image.#####')
    test_X, test_Y = preprocess(origtest_X, origtest_Y)

    print('#####Start Simulate.#####')
    test_output_Y = bp.simulate(test_X[0:50])

    print('#####Check Correct Rate.#####')
    output_Y = postprocess(test_output_Y)
    correctRate = measureCorrectRate(output_Y, origtest_Y[0:50])
    print('Correct Rate of Classifier: {}'.format(correctRate))


if __name__ == '__main__':
    main()

