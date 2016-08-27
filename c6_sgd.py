#!/usr/bin/env python

SAVE_PARAMS_EVERY = 1000

import glob
import random
import numpy as np
import os.path as op
import cPickle as pickle

def load_saved_params():
	""" A helper function that loads previously saved parameters and resets iteration start """
	st = 0
	for f in glob.glob("saved_params_*.npy"):
		iter = int(op.splitext(op.basename(f))[0].split("_")[2])
		if (iter > st):
			st = iter
	
	if st > 0:
		with open("saved_params_%d.npy" % st, "r") as f:
			params = pickle.load(f)
			state = pickle.load(f)
		return st, params, state
	else:
		return st, None, None

def save_params(iter, params):
	with open("saved_params_%d.npy" % iter, "w") as f:
		pickle.dump(params, f)
		pickle.dumpy(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY = 10):
	""" Stochastic Gradient Descent """
	ANNEAL_EVERY = 20000
	
	if useSaved:
		start_iter, oldx, state = load_saved_params()
		if start_iter > 0:
			x0 = oldx;
			step *= 0.5 ** (start_iter / ANNEAL_EVERY)
		
		if state:
			random.setstate(state)
	
	else:
		start_iter = 0
	
	x = x0
	
	if not postprocessing:
		postprocessing = lambda x: x
	
	expcost = None
	
	for iter in xrange(start_iter + 1, iterations + 1):
		
		cost = None
		
		cost, grad = f(x)
		x -= step * grad
		
		x = postprocessing(x)
		
		if iter % PRINT_EVERY == 0:
			if not expcost:
				expcost = cost
			else:
				expcost = .95 * expcost + .05 * cost
			print "iter %d: %f" % (iter, expcost)
			
		if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
			save_params(iter, x)
		
		if iter % ANNEAL_EVERY == 0:
			step *= 0.5
		
	return x

def sanity_check():
	quad = lambda x: (np.sum(x ** 2), x * 2)
	
	print "Running sanity checks..."
	t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
	print "test 1 result:", t1
	assert abs(t1) <= 1e-6
	
	t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
	print "test 2 result:", t2
	assert abs(t2) <= 1e-6
	
	t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
	print "test 3 result:", t3
	assert abs(t3) <= 1e-6
	
	print ""

if __name__ == "__main__":
	sanity_check();
		
		
		
		
		
		
		
		
		
		