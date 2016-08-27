#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

def softmax(x):
											#x=np.array([[1001,1002],[3,4]])
	if len(x.shape) > 1:					#x.shape=(2, 2)  len(x.shape)=2				
		tmp = np.max(x, axis = 1)			#np.max(x, axis = 1)=array([1002,  4])， max in each row
		x -= tmp.reshape((x.shape[0], 1))	#tmp.reshape((x.shape[0], 1))， tmp becomes 2row1column
		x = np.exp(x)						#xi - max this row, then exp
		tmp = np.sum(x, axis = 1)			#array([ 1.36787944,  1.36787944])，sum of each row
		x /= tmp.reshape((x.shape[0], 1))	#xi / sum this row
	
	else:									#x=[1,2]   x.shape=(2,)   len(x.shape)=1
		tmp = np.max(x)
		x -= tmp
		x = np.exp(x)
		tmp = np.sum(x)
		x /= tmp
	
	return x

def test_softmax_basic():
	print "Running basic tests..."
	test1 = softmax(np.array([1,2]))
	print test1
	assert np.amax(np.fabs(test1 - np.array(
		[0.26894142,  0.73105858 ]))) <= 1e-6
	
	test2 = softmax(np.array([[1001,1002],[3,4]]))
	print test2
	assert np.amax(np.fabs(test2 - np.array(
		[[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6
	
	print "You should verify these results!\n"

if __name__ == "__main__":
	test_softmax_basic()