#!/usr/bin/env python

import numpy as np
import random

from c1_softmax import softmax
from c2_gradcheck import gradcheck_naive
from c3_sigmoid import sigmoid, sigmoid_grad


#Each element is divided by square root of square sum of relative row
def normalizeRows(x):

	N = x.shape[0]
	x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30			
	
	return x


def test_normalize_rows():
	print "Testing normalizeRows..."
	x = normalizeRows(np.array([[3.0, 4.0],[1, 2]]))
	print x
	assert (np.amax(np.fabs(x - np.array([[0.6,0.8],[0.4472136,0.89442719]]))) <= 1e-6)
	print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
	""" Softmax cost function for word2vec models """
	
	probabilities = softmax(predicted.dot(outputVectors.T))			#难怪我看不懂，predicted.dot(outputVectors.T) 这个没懂啥意思
	cost = -np.log(probabilities[target])
	
	delta = probabilities
	delta[target] -= 1
	
	N = delta.shape[0]												#delta.shape = (5,)
	D = predicted.shape[0]											#predicted.shape = (3,)
	grad = delta.reshape((N, 1)) * predicted.reshape((1, D))
	gradPred = (delta.reshape((1, N)).dot(outputVectors)).flatten()
	
	return cost, gradPred, grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
	""" Negative sampling cost function for word2vec models """
	
	grad = np.zeros(outputVectors.shape)
	gradPred = np.zeros(predicted.shape)
	
	indices = [target]
	for k in xrange(K):
		newidx = dataset.sampleTokenIdx()
		while newidx == target:
			newidx = dataset.sampleTokenIdx()
		indices += [newidx]
	
	labels = np.array([1] + [-1 for k in xrange(K)])
	vecs = outputVectors[indices, :]
	
	t = sigmoid(vecs.dot(predicted) * labels)
	cost = -np.sum(np.log(t))
	
	delta = labels * (t-1)
	gradPred = delta.reshape((1, K+1)).dot(vecs).flatten()
	gradtemp = delta.reshape((K+1, 1)).dot(predicted.reshape(1, predicted.shape[0]))
	
	for k in xrange(K+1):
		grad[indices[k]] += gradtemp[k, :]
		
	return cost, gradPred, grad
	

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
	dataset, word2vecCostAndGradient = softmaxCostAndGradient):
	""" Skip-gram model in word2vec """
	
	currentI = tokens[currentWord]						#the order of this center word in the whole vocabulary
	predicted = inputVectors[currentI, :]				#turn this word to vector representation
	
	cost = 0.0
	gradIn = np.zeros(inputVectors.shape)
	gradOut = np.zeros(outputVectors.shape)
	for cwd in contextWords:							#contextWords is of 2C length
		idx = tokens[cwd]
		cc, gp, gg = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
		cost += cc										#final cost/gradient is the 'sum' of result calculated by each word in context
		gradOut += gg
		gradIn[currentI, :] += gp
	
	return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
	dataset, word2vecCostAndGradient = softmaxCostAndGradient):
	""" CBOW model in word2vec """
	
	cost = 0
	gradIn = np.zeros(inputVectors.shape)
	gradOut = np.zeros(outputVectors.shape)
	
	D = inputVectors.shape[1]
	predicted = np.zeros((D, ))
	
	indices = [tokens[cwd] for cwd in contextWords]
	for idx in indices:
		predicted += inputVectors[idx, :]
	
	cost, gp, gradOut = word2vecCostAndGradient(predicted, tokens[currentWord], outputVectors, dataset)
	gradIn = np.zeros(inputVectors.shape)
	for idx in indices:
		gradIn[idx, :] += gp
	
	return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
	batchsize = 50
	cost = 0.0
	grad = np.zeros(wordVectors.shape)   #each element in wordVectors has a gradient
	N = wordVectors.shape[0]
	inputVectors = wordVectors[:N/2, :]
	outputVectors = wordVectors[N/2:, :]
	for i in xrange(batchsize):									#train word2vecModel for 50 times
		C1 = random.randint(1, C)
		centerword, context = dataset.getRandomContext(C1)		#randomly choose 1 word, and generate a context of it
		
		if word2vecModel == skipgram:
			denom = 1
		else:
			denom = 1
		
		c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
		cost += c / batchsize / denom							#calculate the average
		grad[:N/2, :] += gin / batchsize / denom
		grad[N/2:, :] += gout / batchsize / denom
	
	return cost, grad			#在run里，sgd返回的是wordvectors，但sgd返回的东西是由wrapper决定的，难道wrapper的gra就是那个wordvectors吗
								#应该是的，W1 W2 就是词向量的矩阵，但是我给忘了原理

def test_word2vec():
	
	dataset = type('dummy', (), {})()   #create a dynamic object and then add attributes to it
	def dummySampleTokenIdx():			#generate 1 integer between (0,4)
		return random.randint(0, 4)
	
	def getRandomContext(C):							#getRandomContext(3) = ('d', ['d', 'd', 'd', 'e', 'a', 'd'])
		tokens = ["a", "b", "c", "d", "e"]
		return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
			for i in xrange(2*C)]
	
	dataset.sampleTokenIdx = dummySampleTokenIdx		#add two methods to dataset
	dataset.getRandomContext = getRandomContext
	
	random.seed(31415)									
	np.random.seed(9265)								#can be called again to re-seed the generator
	
	#in this test, this wordvectors matrix is randomly generated,
	#but in real training, this matrix is a well trained data
	dummy_vectors = normalizeRows(np.random.randn(10,3))					#generate matrix in shape=(10,3), 
	dummy_tokens = dict([("a",0), ("b",1), ("c",2), ("d",3), ("e",4)])		#{'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
	print "==== Gradient check for skip-gram ===="
	gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)  #vec is dummy_vectors
	gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
	print "\n==== Gradient check for CBOW      ===="
	gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)	
	gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
	
	print "\n=== Results ==="
	print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
	print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset, negSamplingCostAndGradient)
	print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
	print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
	
if __name__ == "__main__":
	test_normalize_rows()	
	test_word2vec()


	
	












	
