from math import log
import numpy as np
from time import time
from collections import Counter
import sys

class Tree:
	leaf = True	# boolean if it is a final leaf of a tree or not
	prediction = None	# what is the prediction (if leaf)
	feature = None		# which feature to split on?
	threshold = None	# what threshold to split on?
	left = None			# left subtree
	right = None		# right subtree

class Data:
	feature1 = []
	feature2 = []
	feature3 = []
	feature4 = []	# list of lists (size: number_of_examples x number_of_features)
	labels = []	# list of strings (lenght: number_of_examples)

###################################################################

def read_data(txt_path):
	# TODO: function that will read the .txt file and store it in the data structure
	# use the Data class defined above to store the information
	data = Data()
	training_file = open(txt_path,'r')
	traininglist = days_file.readlines()
	for i in traininglist:
		line_in_traininglist = traininglist[i].split(",")
		feature1.append(line_in_traininglist[0])
		feature2.append(line_in_traininglist[1])
		feature3.append(line_in_traininglist[2])
		feature4.append(line_in_traininglist[3])
		labels.append(line_in_traininglist[4])

	return data

def predict(tree, point):
	# TODO: function that should return a prediction for the specific point (predicted label)
	if tree.leaf = false:
		if point[tree.feature] < tree.threshold:
			return predict(tree.left, point)
		elif point[tree.feature] >= tree.threshold:
			return predict(tree.right, point)
	prediction = tree.prediction

	return prediction

def split_data(data, feature, threshold):
	# TODO: function that given specific feature and the threshold will divide the data into two parts
	leftdata = Data()
	rightdata = Data()
	if feature == 1:
		flist = data.feature1
	elif feature == 2:
		flist = data.feature2
	elif feature == 3:
		flist = data.feature3
	elif feature == 4:
		flist = data.feature4
	for x in flist:
		if flist[x] < split:
			leftdata.labels.append(labels[x])
			leftdata.features1.append(data.feature1[x])
			leftdata.features2.append(data.feature2[x])
			leftdata.features3.append(data.feature3[x])
			leftdata.features4.append(data.feature4[x])
		elif flist[x] > split:
			rightdata.labels.append(labels[x])
			rightdata.features1.append(data.feature1[x])
			rightdata.features2.append(data.feature2[x])
			rightdata.features3.append(data.feature3[x])
			rightdata.features4.append(data.feature4[x])
		elif flist[x] == split:
			rightdata.labels.append(labels[x])
			rightdata.features1.append(data.feature1[x])
			rightdata.features2.append(data.feature2[x])
			rightdata.features3.append(data.feature3[x])
			rightdata.features4.append(data.feature4[x])
	return (leftdata, rightdata)

def get_entropy(data):
	# TODO: calculate entropy given data
	versicolor = 0
	setosa = 0
	virginica = 0
	total = 0
	for i in data.labels:
			if data.labels[i] == "Iris-versicolor":
				versicolor+=1
			elif data.labels[i] == "Iris-setosa":
				setosa+=1
			elif data.labels[i] == "Iris-viginica":
				virginica+=1
	total = versicolor + setosa + virginica
	entropy = -(versicolor/total*math.log2(versicolor/total) + setosa/total*math.log2(setosa/total) + virginica/total*math.log2(virginica/total))
	return entropy

def find_best_threshold(data, feature):
	# TODO: iterate through data (along a single feature) to find best threshold (for a specified feature)
	best_gain = 0
	if feature == 1:
		flist = data.feature1
	elif feature == 2:
		flist = data.feature2
	elif feature == 3:
		flist = data.feature3
	elif feature == 4:
		flist = data.feature4
	for i in flist:
		split = flist[i]
		split1, split2 = split_data(data, feature, split)
		entropy1 = get_entropy(split1)
		entropy2 = get_entropy(split2)
		entropytot = get_entropy(data.labels)

		total = len(split1) + len(split2)
		gain = entropytot - (len(split1)/total*entropy1 + len(split2)/total*entropy2)

		if gain > best_gain:
			best_gain = gain
			best_threshold = split

		

	return best_gain, best_threshold

def find_best_split(data):
	# TODO: iterate through data along all features to find the best possible split overall
	best_gain, best_threshold = find_best_threshold(data, 1)
	for i in range(4):
		gain, threshold = find_best_threshold(data, i+1)
		if gain > best_gain:
			best_gain = gain
			best_threshold = threshold
			best_feature = i
	return best_feature, best_threshold


def c45(data):
	# TODO: Construct a decision tree with the data and return it.
	tree = Tree()
	tree = c45helper(data, tree)
	return tree

def c45helper(data, treein):
	feature, threshold = find_best_split(data)
	treein.feature = feature
	treein.threshold = threshold
	gain, gainthreshold = find_best_threshold(data, feature)
	if gain < 0.1:
		treein.leaf = True
		treein.prediction = getPrediction(data)
		pass
	left, right = split_data(data, feature, threshold)
	tree.left = c45helper(left, treein)
	tree.right = c45helper(right, treein)
	return treein

def getPrediction(data):
	for i in data.labels:
			if data.labels[i] == "Iris-versicolor":
				versicolor+=1
			elif data.labels[i] == "Iris-setosa":
				setosa+=1
			elif data.labels[i] == "Iris-viginica":
				virginica+=1
	total = versicolor + setosa + virginica
	if versicolor/total > setosa/total:
		if versicolor/total > virginica total:
			return versicolor
	if setosa/total > versicolor/total:
		if setosa/total > virginica/total:
			return setosa
	if virginica/total > versicolor/total:
		if virginica/total > setosa/total:
			return virginica


def test(data, tree):
	# TODO: given data and a constructed tree - return a list of strings (predicted label for every example in the data)
	for i in data.feature:
		point[0] = data.feature1[i]
		point[1] = data.feature2[i]
		point[2] = data.feature3[i]
		point[3] = data.feature4[i]
		point[4] = data.labels[i]
		predictions.append(prediction(tree, point))
	return predictions
###################################################################


