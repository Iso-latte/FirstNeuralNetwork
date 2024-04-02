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
    self.y = y
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

  def test(self, X):
    '''
    :param A singular or collection of 3x1 arrays
    :type Numpy array
    :returns predicted output of the arrays 2x1 arrays
    '''
    self.layer1 = sigmoid(np.dot(X, self.weights1))
    self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
    return self.layer2

''' NAMESPACE ENTRY POINT FOR THE CODE '''

if __name__ == '__main__':

  # loss list used for plotting loss
  loss_list = []

  #iteration list used for plotting loss
  iteration_list = []

  # Input layer 3
  X=np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0]), dtype=float)

  # Output layer 2
  y=np.array(([0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]), dtype=float)

  # Initializing Neural Network object
  NN = NeuralNetwork(X,y)

  for i in range(1000): # trains the NN 1,000 times
    NN.train(X, y)
    loss_list.append((np.mean(np.square(y - NN.feedforward()))))  # adding the loss of the model into list for graphing
    iteration_list.append(i)  # adding the iteration number into a list of graphing

  print("\n\n"+"*"*20+" - REPORT - "+"*"*20+"\n\n")

  print("*** THE DIMENSIONS OF WEIGHT MATRICES ***\n")
  # Printing shape of weight matrix one
  print("- The dimension of weight matrix one: \n")
  print('\t'+str(NN.weights1.shape)+'\n\n')
  # Printing shape of weight matrix two
  print("- The dimension of weight matrix two: \n")
  print('\t'+str(NN.weights2.shape)+'\n\n')


  # Printing the test of sample [1,1,1]
  print("*** Testing sample X1 [1,1,1]: ***\n")
  # Initializin a new array with data
  X1 = np.array(([1,1,1]), dtype=float)
  test_pred = NN.test(X1)  # Predicting the value for given array
  round_pred = np.round(test_pred)  # Rounding the prediction for the test prediction
  print("\t - Predicted Output: "+str(test_pred)+"\n")
  print("\t - Predicted Output (rounded): " + str(round_pred)+"\n\n")


  # Printing out the matrix for loss over iterations
  fig, ax = plt.subplots()
  ax.plot(iteration_list, loss_list)
  ax.set(xlabel='Number of Iterations', ylabel='Loss (Mean Sum Squared Loss)',title='Change in Loss by Iteration')
  plt.show()

  # Print out the martix for loss over iterations with data that further shows the fine change of loss
  fig, ax = plt.subplots()
  ax.plot(iteration_list[100:], loss_list[100:])
  ax.set(xlabel='Number of Iterations', ylabel='Loss (Mean Sum Squared Loss)',title='Change in Loss by Iteration (Excluding first 100 iterations)')
  plt.show()