import handwriteRecognition
import unittest
import matplotlib.pyplot as plt
from numpy import *
from random import uniform

class handwriteRecognitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = \
                                     handwriteRecognition.load()

    @classmethod
    def tearDownClass(self):
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None


    def test_load(self):
        plt.figure()
        plt.subplot(121)
        plt.imshow(array(self.train_X[0]).reshape(28, 28), cmap='gray')
        plt.subplot(122)
        plt.imshow(array(self.test_X[0]).reshape(28, 28), cmap='gray')
        plt.show()
        self.assertEqual(len(self.train_X), len(self.train_Y))
        self.assertEqual(len(self.test_X), len(self.test_Y))


    def test_extractPixelMapFeature(self):
        X0 = handwriteRecognition.extractPixelMapFeature(self.train_X[0])
        X1 = handwriteRecognition.extractPixelMapFeature(self.test_X[0])
        plt.figure()
        plt.subplot(121)
        plt.imshow(array(X0).reshape(28, 28), cmap='gray')
        plt.subplot(122)
        plt.imshow(array(X1).reshape(28, 28), cmap='gray')
        plt.show()
        self.assertTrue(set(X0) <= {0, 1, 2, 3})
        self.assertTrue(set(X1)  <= {0, 1, 2, 3})


    def test_encodeLabel(self):
        for iter in range(10):
            Y = handwriteRecognition.encodeLabel((iter,), 10)
            resultY = [0 for i in range(10)]
            resultY[iter] = 1
            self.assertEqual(Y, tuple(resultY))


    def test_preprocess_and_postprocess(self):
        X0, Y0 = handwriteRecognition.preprocess(
                    self.train_X[0: 10],
                    self.train_Y[0: 10], 10)
        X1, Y1 = handwriteRecognition.preprocess(
                    self.test_X[0: 10],
                    self.test_Y[0: 10], 10)
        self.assertEqual(len(X0), 10)
        self.assertEqual(len(Y0), 10)
        self.assertEqual(len(X1), 10)
        self.assertEqual(len(Y1), 10)

        resultY0 = handwriteRecognition.postprocess(Y0, 10)
        resultY1 = handwriteRecognition.postprocess(Y1, 10)
        self.assertEqual(Y0[0: 10], resultY0)
        self.assertEqual(Y1[0: 10], resultY1)


    def test_decodeLabel(self):
        for iter in range(10):
            Y = [uniform(0, 0.5) for i in range(10)]
            Y[iter] = uniform(0.5, 1)
            Y = handwriteRecognition.decodeLabel(Y, 10)
            resultY = [0 for i in range(10)]
            resultY[iter] = 1
            self.assertEqual(Y, tuple(resultY))


    def test_measureCorrectRate(self):
        testRate = handwriteRecognition.measureCorrectRate(self.test_Y[0: 10],
                                                           self.test_Y[0: 10])
        self.assertEqual(1, testRate)



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(handwriteRecognitionTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
