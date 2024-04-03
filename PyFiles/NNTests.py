from NeuralNetwork import NeuralNetwork
import numpy as np
import unittest

class NNtests(unittest.TestCase):

    def setUp(self):
        self.num = 5000
        X=np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0]), dtype=float)
        y=np.array(([0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]), dtype=float)
        self.NN = NeuralNetwork(X,y)
        self.NN.train_iterations(self.num)

    def test_iterations(self):
        self.NN.train_iterations(0)
        self.assertEqual(self.NN.iterations, 0)
        self.NN.train_iterations(self.num)
        self.assertEqual(self.NN.iterations, self.num)
    
    def test_loss(self):
        self.assertLess(self.NN.get_loss(),0.01)
    
    def test_weight_matrices_shape(self):
        self.assertEqual(self.NN.weights1.shape, (3,6))
        self.assertEqual(self.NN.weights2.shape, (6,2))
    
    def test_predictions(self):
        X1 = np.array(([1,1,1]), dtype=float)
        pred_array = self.NN.predict(X1)
        self.assertGreater(pred_array[0], 0.90)
        self.assertLess(pred_array[1], 0.01)
        rounded_array = self.NN.predict_round
        self.assertEqual(rounded_array[0],1)
        self.assertEqual(rounded_array[1],0)



if __name__ == '__main__':
    unittest.main()