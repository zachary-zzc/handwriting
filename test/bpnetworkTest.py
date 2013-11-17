import sys
sys.path.append('..')

import os
from numpy import *
from random import uniform
from utils.plot import plot_cost, plot_regression
import unittest
import bpnetwork
import gc


class bpNetworkTest(unittest.TestCase):

    bps_edge_test_set_x = zeros((10, 1))
    bps_edge_test_set_t = ones((10, 1))
    w0 = array([[0, 0, 0],
                [uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)]])
    w1 = array([[1],
                [uniform(-1, 1)],
                [uniform(-1, 1)],
                [uniform(-1, 1)]])

    bps_test_set_x = array([[uniform(-1, 1)] for i in range(50)])
    bps_test_set_t = []
    bps_test_set_t.append(sin(bps_test_set_x))
    bps_test_set_t.append(bps_test_set_x ** 2)
    bps_test_set_t.append(abs(bps_test_set_x))
    #bps_test_set_t.append(step_func(bps_test_set_x))


    def setUp(self):
        self.testname = "test"
        hidden_layer_num = 1
        hidden_unit_list = (3, )
        acfunc_list = ('tanh', )
        output_func = 'line'
        sigma = 1e-2
        steps = 1000
        self.bps = bpnetwork.bpNetWork(hidden_layer_num,
                                              hidden_unit_list,
                                              acfunc_list,
                                              output_func,
                                              sigma,
                                              steps)


    def tearDown(self):
        gc.enable()
        gc.set_debug(gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE)
        print('garbage object num: {}'.format(len(gc.garbage)))
        del self.bps


    def test_bpnetwork_initw(self):
        self.bps.initw(self.bps_edge_test_set_x, self.bps_edge_test_set_t)
        self.assertEqual(self.bps.input_layer.w.shape, (2, 3))
        self.assertEqual(self.bps.hidden_layers[0].w.shape, (4, 1))
        self.testname = 'init';


    def test_bpnetwork_forward_propagation(self):
        self.bps.initw(self.bps_edge_test_set_x, self.bps_edge_test_set_t)
        self.bps.input_layer.setw(self.w0)
        self.bps.hidden_layers[0].setw(self.w1)
        self.bps.forward_propagation(self.bps_edge_test_set_x)
        self.assertTrue((self.bps.Y == self.bps_edge_test_set_t).all())
        self.testname = 'forward propagation'


    def test_bpnetwork_backward_propagation(self):
        self.bps.initw(self.bps_edge_test_set_x, self.bps_edge_test_set_t)
        self.bps.input_layer.setw(self.w0)
        self.bps.hidden_layers[0].setw(self.w1)
        self.bps.forward_propagation(self.bps_edge_test_set_x)
        self.bps.backward_propagation(self.bps_edge_test_set_t)
        self.assertTrue((self.bps.delta_list[0] == zeros((10, 3))).all())
        self.assertTrue((self.bps.delta_list[1] == zeros((10, 1))).all())
        self.testname = 'backward propagation'


    def test_bpnetwork_cost(self):
        self.bps.initw(self.bps_edge_test_set_x, self.bps_edge_test_set_t)
        self.bps.input_layer.setw(self.w0)
        self.bps.hidden_layers[0].setw(self.w1)
        self.bps.forward_propagation(self.bps_edge_test_set_x)
        cost = self.bps.costFunc(self.bps_edge_test_set_t)
        self.assertEqual(cost, 0)
        self.testname = 'cost'


    def test_bpnetwork_gradient_check(self):
        pass


    def test_bpnetwork_train(self):
        steps = linspace(-1, 1, 100)
        for iter in range(len(self.bps_test_set_t)):
            self.bps.train(self.bps_test_set_x, self.bps_test_set_t[iter])
            self.assertAlmostEqual(self.bps.cost, 0, places=1)
            regression = self.bps.simulate(steps)
            plot_regression(self.bps_test_set_x,
                            self.bps_test_set_t[iter],
                            steps,
                            regression
                            )
            plot_cost(self.bps.debug_x, self.bps.debug_y)
        self.testname = 'train'


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(bpNetworkTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
