from NeuralNetwork import NeuralNetwork
import numpy as np
import unittest

class NNtests(unittest.TestCase):

    def test_iterations(self):
        X=np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0]), dtype=float)
        y=np.array(([0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]), dtype=float)
        NN = NeuralNetwork(X,y)
        self.assertEqual(NN.iterations, 0)
        NN.train_iterations(1000)
        self.assertEqual(NN.iterations, 1000)


if __name__ == '__main__':
    unittest.main()