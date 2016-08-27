#!/usr/bin/env python

import numpy as np
import random

# Function: 
# for each element in x
# compare derivative calculated by formular and calculus
# f: 1st parameter is cost function, 2nd parameter is gradient
def gradcheck_naive(f, x):
	
	#Return an object capturing the current internal state of the generator
	rndstate = random.getstate()			#why use state??????
	random.setstate(rndstate)
	fx, grad = f(x)							#fx=np.sum(x ** 2), grad=x * 2 
	h = 1e-4
	
	#Efficient multi-dimensional iterator object to iterate over arrays
	# Iterate over all indexes in x
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])	
	
	while not it.finished:
		ix = it.multi_index					#starts from (0, 0) then (0, 1)
		
		x[ix] += h							#To calculate [f(xi+h)-f(xi-h)] / 2h
		random.setstate(rndstate)
		fxh, _ = f(x)
		x[ix] -= 2*h
		random.setstate(rndstate)
		fxnh, _ = f(x)
		x[ix] += h
		numgrad = (fxh - fxnh) / 2 / h
											#To compare gradient calculated by formular and calculus
		reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
		if reldiff > 1e-5:
			print "Gradient check failed."
			print "First gradient error found at index %s" % str(ix)
			print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
			return
		
		it.iternext()
		
	print "Gradient check passed"

def sanity_check():
	
	quad = lambda x: (np.sum(x ** 2), x * 2)
	
	print "Running sanity checks..."
	gradcheck_naive(quad, np.array(123.456))
	gradcheck_naive(quad, np.random.randn(3,))
	gradcheck_naive(quad, np.random.randn(2,3))
	print ""

if __name__ == "__main__":
	sanity_check()