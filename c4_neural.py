#!/usr/bin/env python

import numpy as np
import random

from c1_softmax import softmax
from c3_sigmoid import sigmoid, sigmoid_grad
from c2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):

	# Distribute elements in params to W1 b1 W2 b2
	ofs = 0
	Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
	
	W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))					#W1.shape = (10, 5)
	ofs += Dx * H
	b1 = np.reshape(params[ofs:ofs + H], (1, H))						#b1.shape = (1, 5)
	ofs += H
	W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))					#W2.shape = (5, 10)
	ofs += H * Dy
	b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))						#b2.shape = (1, 10)
	
	# Forward Propagation
	hidden = sigmoid(data.dot(W1) + b1)									#hidden.shape = (20, 5)
	prediction = softmax(hidden.dot(W2) + b2)							#prediction.shape = (20, 10)
	cost = -np.sum(np.log(prediction) * labels)							#np.log(prediction) * labels remains relative class value
	
	# Backward Propagation
	delta = prediction - labels
	gradW2 = hidden.T.dot(delta)
	gradb2 = np.sum(delta, axis = 0)
	
	delta = delta.dot(W2.T) * sigmoid_grad(hidden)
	gradW1 = data.T.dot(delta)
	gradb1 = np.sum(delta, axis = 0)
	
	grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),			#grad.shape = (115,)  concatenate 4 gradients together
		gradW2.flatten(), gradb2.flatten()))
	
	# f: 1st parameter is cost function, 2nd parameter is gradient	
	# gradient is of params
	return cost, grad

def sanity_check():
	
	print "Running sanity check..."
	
	N = 20												
	dimensions = [10, 5, 10]							#Dimension of x, H, y=labels
	data = np.random.randn(N, dimensions[0])			#data.shape, labels.shape = (20, 10)
	labels = np.zeros((N, dimensions[2]))					#Data has 10 columns = 10 input x vectors
	for i in xrange(N):										#Data has 20 rows = each x has 20 features
		labels[i, random.randint(0, dimensions[2]-1)] = 1	#each row randomly set a position to 1
															#each vector xi belongs to 1 class
															
	params = np.random.randn((dimensions[0]+1) * dimensions[1] + 
		(dimensions[1]+1) * dimensions[2], )				#params.shape = (115,)
	
	# For each element in params, compare derivative calculated by calculus and BP-NN
	gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)
	

if __name__ == "__main__":
	sanity_check()
	
	
	
	
	
