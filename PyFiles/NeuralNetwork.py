import numpy as np
import matplotlib.pyplot as plt

# Activation function
def sigmoid(t):
  return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
  return p * (1 - p)


class NeuralNetwork:
  def __init__(self, x, y):
    '''
    :param two arrays a input layer "x" and output layer "y"
    :type Numpy array
    '''
    self.input = x
    self.y = y
    self.iterations = 0  # This is so you can print out the info on the model
    self.output = np.zeros(y.shape)
    self.weights1= np.random.rand(self.input.shape[1],6)  # This is so the hidden layer has 6 nodes
    self.weights2 = np.random.rand(6,2)  # This is so the hidden layer has six nodes in hidden layer and 2 in output

  def feedforward(self):
    '''
    :returns the predicted outcome of the data in 2x1 matices
    '''
    self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
    return self.layer2

  def backprop(self):
    '''
    :function performs backpropagation
    '''
    d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output)*sigmoid_derivative(self.output))
    d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output),self.weights2.T)*sigmoid_derivative(self.layer1))
    self.weights1 += d_weights1
    self.weights2 += d_weights2

  def train(self, X, y):
    '''
    :function uses class methods backprop() and feedforward() to train NN
    '''
    self.output = self.feedforward()
    self.backprop()

  def predict(self, X):
    '''
    :param A singular or collection of 3x1 arrays
    :type Numpy array
    :returns predicted output of the arrays 2x1 arrays
    '''
    self.layer1 = sigmoid(np.dot(X, self.weights1))
    self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
    return self.layer2
  
  def train_iterations(self, num):
    '''
    :param An integer for number of iterations wanted for the training of the NN
    :type Integer
    '''
    self.iterations = num  # setting attribute iterations
    for i in range(num): # trains the NN 1,000 times
      self.train(self.input, self.y)

  def get_iterations(self):
    '''
    :returns the number of iterations
    :type Integer
    '''
    return self.iterations
  
  def get_loss(self):
    '''
    :returns the loss of the model
    :type Integer
    '''
    return np.mean(np.square(self.y - self.feedforward()))
  
  def info(self):
    '''
    :returns a string that gives info on the training and loss of the model
    '''
    info_string = "\n Iterations trained for: " + str(self.get_iterations()) + "\n Loss of model: " + str(self.get_loss()) + '\n'
    return info_string