from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

def NNExample_main():
  
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
  test_pred = NN.predict(X1)  # Predicting the value for given array
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

''' NAMESPACE ENTRY POINT FOR THE CODE '''

if __name__ == '__main__':
    NNExample_main()