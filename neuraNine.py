import pandas as pd
import numpy as np
from PIL import Image
import random 
# import matplotlib.pyplot as plt

test_data = pd.read_csv("D:\Python Directory\Mnist_data\mnist_test.csv")
test_data = np.array(test_data) # converted data to array
np.random.shuffle(test_data) # shuffled data randomly
data = test_data.T # transposing the array
# now has 785 rows and 10000 columns
# now each column is having data of one training eg. where 1st value is the "label"
y_test = data[0,] ;
x_test = data[1:,] / 255 ;
y_test=y_test.reshape(y_test.shape[0],1)
y_test=y_test.T

# initialize parameters for deep neural networks
def initialize_parameters_deep(layer_dims):
    # Assuming your CSV loading and processing remains the same
    D = pd.read_csv("D:\Python Directory\Mnist_data\AWB.csv", header=None)
    D = np.array(D)
    keys = D[:, :1]
    D = D[:, 1:]
    parameters = {}
    L = len(layer_dims)
    i=-1;
    for l in range(1, L):
        i+=1;
        W_data = D[i, 0:layer_dims[l] * layer_dims[l-1]].reshape(layer_dims[l], layer_dims[l-1])
        i+=1;
        b_data = D[i, 0:layer_dims[l]].reshape(layer_dims[l], 1)
        parameters['W' + str(l)] = np.array(W_data, dtype=np.float64)
        parameters['b' + str(l)] = np.array(b_data, dtype=np.float64)
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b): 
    # A is the activation which is actually X or X_train
    # Weight is of a single layer eg W1 and same for bias b1
    # A with shape (784, 60000) & W with shape (60, 784) will give Z with shape (60,m000)
    # (60,60000) this shape means there is a Z for each neuron in current layer and for all training eg.
    # 60 - neurons in current layer & 60000 is no of training eg.
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    # ensures to maintain the dimension
    cache = (A, W, b) 
    # cache is a tuple containing all 3 ie. A,W,b
    
    return Z, cache

# useful activation functions and their derivatives
def sigmoid_(Z):
    return 1/(1+np.exp(-Z))

def relu_(Z):
    return Z*(Z>0)

def drelu_(Z):
    return 1. *(Z>0)

def dsigmoid_(Z):
    return sigmoid_(Z)*(1-sigmoid_(Z))

def sigmoid(Z):
    return sigmoid_(Z),Z

def relu(Z):
    return relu_(Z),Z

def linear_activation_forward(A_prev,W,b,activation):
    global i;
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b) # so here we get Z and a tuple having (A,W,b)
        A, activation_cache = sigmoid(Z) # we pass Z to sigmoid() t get A and Z named as activation_cache
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b) # so here we get Z and a tuple having (A,W,b)
        A, activation_cache = relu(Z) # we pass Z to relu() t get A and Z named as activation_cache
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    # so shape of W is (60,784) and A is (784,60000)
    # so we get shape of A = (60,60000) ie. the new values of activtion for all neurons of current layer and for all eg.
    cache = (linear_cache, activation_cache)
    # linear_cache -> (A,W,b)
    # activation_cache -> Z
    
    # so we are returning 'A' and ( (A_prev,W,b) , (Z) ) for a single layer but for all eg.
    return A, cache

# implementation of forward propogation for L layer neural network
def L_model_forward(X, parameters):
    caches = []
    A = X # dim of X is (784,60000)
    L = len(parameters) // 2   # = 3 here
    # operator // means floorDivision should give 3 here as parameters -> {W1,b1,W2,b2,W3,b3}
    # so inside the loop W1,b1,W2,b2 are utilised
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    caches.append(cache)
    #assert(AL.shape == (1,X.shape[1]))
    # dim of AL is (10,60000) ie. the activation of output layer for all eg.
    return AL, caches

parameters = initialize_parameters_deep([784,60,10,10])
def random_test():
    index = random.randint(0,9999)
    sample = [row[index] for row in x_test]
    sample = np.array(sample)
    # print(sample);
    # array_to_image(sample)
    # display_image_in_terminal(sample)
    sample = sample.reshape(sample.shape[0],1)
    # sample = get_from_csv();
    A_out,caches = L_model_forward(sample,parameters)
    Output = np.argmax(A_out)
    print('guess',Output,', actual',y_test[0][index])

# convert img to csv
def array_to_image(array):
    img_array = array.reshape(28, 28)
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, 'L')  # 'L' mode is for grayscale images
    img.show()

# def display_image_in_terminal(array):
#     img_array = array.reshape(28, 28)
#     img_array = (img_array * 255).astype(np.uint8)
#     plt.imshow(img_array, cmap='gray') 
#     plt.axis('off')
#     plt.show()

def get_from_csv(path): #get img array from csv to test
    img = pd.read_csv(path)
    img = np.array(img, dtype = np.float64)
    print(img.shape)
    img = img.reshape(785,1)
    return img

random_test();
