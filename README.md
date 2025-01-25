# DigitRecognition
Python code from scratch, without any ML libraries to recognize handwritten digits from scratch.


Artificial Neural Network is more interesting than what it just seems to be. The new world of inventions and innovation, all the possibilities opened up due to a simple arrangement of neurons in layers is stunning. The scope even widens up when the internal structure of the ANN or CNN is untangled. The neural network designed here from scratch displayed it's accuracy on training data to be 99.87\% and 98.88\% on testing data. A deeper understanding of the neural network, its structure and its working enables one to formulate and design a neural network according to the requirements, which is here applied to it's training  process. Recognising handwritten digits implemented here is one of the simplest and most effective task a neural network can be designed for, so as to gain the understanding of the internals of the NN.

The dataset being used for this model to be trained and tested is the popular MNIST dataset, which is widely used in field of machine learning. 

The MNIST dataset has the following features:

Size - 70,000 images.
Class count - 10
Training example - 60,000 images.
Testing examples - 10,000 images.
Image resolution - 28*28 pixels.
Labels - 0 to 9


Libraries
NumPy is very useful library provided by Python which makes complex calculation and computation like matrix algebra very convenient and handy. 
This NumPy library is being utilized to reduce the computation cost, keeping in mind not to compromise with the performance and flexibility of the network.

Pandas library provided by python is being utilized to read and write data of the network from and to a CSV file. 
This  use of CSV file along with Pandas enables the neural network to be trained in partitions instead of one go. 

Matplotlib is being utilized to plot the graph of the cost function and the learning curve.
