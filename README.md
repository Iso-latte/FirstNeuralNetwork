# First Neural Network
## Google colab link: https://colab.research.google.com/drive/1YsDhmFcN9i8C3LBlhAMR4ATqwqoA8avX#scrollTo=1XKwZ0nfSsew

### Packages used:
- Numpy and matplotlib

### Class Neural Network:
- Contains 6 class attributes
- Contains 5 class methods

### Info:
- Creates a 3-6-2 Neural Network
- 3 nodes for input layer
- 6 nodes for hidden layer
- 2 nodes for output layer
- Activation function used is a sigmoid
- Uses back propergation to update the weights
- Trains the NN 1,000 times
- Plots the loss over number of training iterations
- Prints out the shape of both weight matrices
- Prints out the predicted value for the test case and the rounded value of the test case

### Questions:
- What are the dimensions of weight matrix 1 and weight matrix 2?
  - The dimension of weight matrix 1 is: 3x6
  - The dimension of weight matrix 2 is: 6x2
- Test [1,1,1]. What are the predicted y values for it?
  - The predicted y values for this sample are: [0.98363427, 0.0157898] (subject to change)
  - The rounded y values for this sample are: [1,0]
- Could you guess what is the meaning of y1 and y2?
  - My guess what the meaning of y1 and y2 mean is that the outcome is [0,1] when the input is even in binary and [1,0] when the outcome is odd in binary. For example 000, 010, 100 are all [0,1] while the sequences 001, 011, 111 are all [1,0]. This makes me believe that this is determining whether the binary sequence is even or odd.
