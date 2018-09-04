#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:09:39 2018

@author: grahamcooke
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def f(x, y):
    return ((x**2 + y**2 < 0.25))

def sigmoid(x, deriv=False):
	if not deriv:
		return 1 / (1 + np.exp (-x))
	else:
		out = sigmoid(x)
		return out * (1 - out)
    
def makeData(size, uniform=False, error=0):
    if uniform:
        side = math.sqrt(size)
        side = round(side)
        x = np.arange(-1, 1, 1/side) + np.zeros((side, 1))
        y = x.T[::-1]
        x = x.ravel()
        y = y.ravel()
        
    else:
        np.random.seed(0)
        x = 2*np.random.random_sample((1,size))-1
        y = 2*np.random.random_sample((1,size))-1
    #z = ((x**2 + y**2 < 0.25) + ((x-0.75)**2 + (y-0.75)**2 < 0.05))*1
    z = f(x,y)
    print(z)
    if error != 0:
        noise = (np.random.random_sample((1,size)) <= error)*1
        print(noise)
        z = (noise + z) * ((1-noise) + (1-z))
    print(z)
    return np.vstack((x,y,z))
    
class neuralNet:
    
    nLayers = 0
    structure = []
    # To clarify here, layers is the activation of the nodes on a layer,
    # while z is the dot product result from the pervious 
    # layer (before going through the sigmoid)
    layers = []
    z = []
    weights = []
    
    def __init__ (self, nodes):
        
        self.nLayers = 0
        self.structure = []
        # To clarify here, layers is the activation of the nodes on a layer,
        # while z is the dot product result from the pervious 
        # layer (before going through the sigmoid)
        self.layers = []
        self.z = []
        self.weights = []
        
        self.nLayers = len(nodes)
        self.structure = nodes
        #all layers after the first
        layerin = nodes[1:]
        #all layers before the last
        layerout = nodes[:-1]
        #pair them up and iterate through the pairs
        np.random.seed(0)
        for (n,m) in zip(layerin,layerout):
            #+1 for the bias
            self.weights.append(2*np.random.random_sample((n,m+1))-1)
    
    def forward(self, data):
        
        #Clean out previous runs
        self.layers = []
        self.z = []
        
        self.layers.append(data)
        #Makes the indecies of z match those of layers
        self.z.append(np.zeros(data.shape))
        
        for i in range(self.nLayers-1):
            self.layers[i] = np.vstack((self.layers[i], np.ones((1,self.layers[i].shape[1]))))
            a = np.dot(self.weights[i],self.layers[i])
            self.z.append(a)
            self.layers.append(sigmoid(self.z[i+1]))
        return self.layers[-1]
            
    def back(self, data, target, alpha=0.2):
        
        self.forward(data)
        
        #Derivatives of the error with respect to the z values
        dz = []
        #Derivatives of the error with respect to the weights
        dweights = []
        
        error = np.sum((self.layers[-1] - target)**2)
        
        dz.append(sigmoid(self.z[-1],True) * 2 * (self.layers[-1] - target))
        
        for i in reversed(range(self.nLayers-1)):
            dweights.append((1/data.shape[0]) * alpha * np.tensordot(dz[-1],self.layers[i].T,1))
            dz.append(sigmoid(self.z[i],True) * np.dot(self.weights[i].T[:-1:],dz[-1]))

            
        dweights = dweights[::-1]
        deriv = 0
        
        for i in range(len(self.weights)):
            self.weights[i] -= dweights[i]
            deriv += (1/dweights[i].size)*np.sum(dweights[i])
        
        return (error, deriv)
        
networks = [2,4,10,20,40,100,200]
    
data = makeData(500, False, 0.1)

batch = 5
data = np.hsplit(data, batch)

for n in networks:
    
    net = neuralNet((2, n, n, 1))
    
    maxIterations = 100000
    minError = 1e-5
    
    for i in range(maxIterations + 1):
        error = net.back(data[i%batch][:-1:], data[i%batch][-1::], 0.1)
        if i % 2500 == 0:
            print("Iteration {0}\tError: {1:0.6f}\tDerivative: {2:0.6f}".format(i,error[0],error[1]))
        if error[0] <= minError:
            print("Minimum error reached at iteration {0}".format(i))
            break
        
    """
    out = net.forward(data[:-1:]).T
    
    
    print('Input \tOutput \t\tTarget')
    for i in range(data.shape[0]):
        print('{0}\t {1} \t{2}'.format(data[i][:-1:], out[i], data[i][-1::]))
    """
        
    x = np.arange(-1, 1, 0.01) + np.zeros((200, 1))
    
    y = x.T[::-1]
    
    x = x.ravel()
    y = y.ravel()
    
    z = net.forward(np.vstack((x,y)))
    
    z = z.reshape(200,200)
    
    fig = plt.figure(figsize=(8,8))
    
    #fig.add_subplot(2, 1, 1)
    result = plt.imshow(z)
    result.set_cmap("hot")
    plt.colorbar()
    plt.savefig("nodes3" + str(n) + ".png", bbox_inches='tight')
    plt.title("Network Result")
    
    '''
    fig.add_subplot(2, 1, 2)
    expected = plt.imshow(target)
    expected.set_cmap("hot")
    plt.colorbar()
    plt.title("Expected Result")
    '''
    #plt.show()
        