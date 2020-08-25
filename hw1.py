from math import log
#import numpy as np
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
	traininglist = training_file.readlines()
	for i in traininglist:
		line_in_traininglist = traininglist[i].split(",")
		data.feature1.append(line_in_traininglist[0])
		data.feature2.append(line_in_traininglist[1])
		data.feature3.append(line_in_traininglist[2])
		data.feature4.append(line_in_traininglist[3])
		data.labels.append(line_in_traininglist[4])

	return data

def predict(tree, point):
	# TODO: function that should return a prediction for the specific point (predicted label)
	if tree.leaf == False:
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
		if flist[x] < threshold:
			leftdata.labels.append(data.labels[x])
			leftdata.feature1.append(data.feature1[x])
			leftdata.feature2.append(data.feature2[x])
			leftdata.feature3.append(data.feature3[x])
			leftdata.feature4.append(data.feature4[x])
		elif flist[x] > threshold:
			rightdata.labels.append(data.labels[x])
			rightdata.feature1.append(data.feature1[x])
			rightdata.feature2.append(data.feature2[x])
			rightdata.feature3.append(data.feature3[x])
			rightdata.feature4.append(data.feature4[x])
		elif flist[x] == threshold:
			rightdata.labels.append(data.labels[x])
			rightdata.feature1.append(data.feature1[x])
			rightdata.feature2.append(data.feature2[x])
			rightdata.feature3.append(data.feature3[x])
			rightdata.feature4.append(data.feature4[x])
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
	entropy = -(versicolor/total*log(versicolor/total, 2) + setosa/total*log(setosa/total, 2) + virginica/total*log(virginica/total, 2))
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
	treein.left = c45helper(left, treein)
	treein.right = c45helper(right, treein)
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
		if versicolor/total > virginica/total:
			return versicolor
	if setosa/total > versicolor/total:
		if setosa/total > virginica/total:
			return setosa
	if virginica/total > versicolor/total:
		if virginica/total > setosa/total:
			return virginica


def test(data, tree):
	# TODO: given data and a constructed tree - return a list of strings (predicted label for every example in the data)
	point = []
	predictions = []
	for i in data.feature:
		point[0] = data.feature1[i]
		point[1] = data.feature2[i]
		point[2] = data.feature3[i]
		point[3] = data.feature4[i]
		point[4] = data.labels[i]
		predictions.append(predict(tree, point))
	return predictions
###################################################################

def main():
	usetree = Tree()
	traininglist = []
	testinglist = []
	trainingdata = Data()
	testingdata = Data()
	trainingdata = read_data("hw1_train.txt")
	testingdata = read_data("hw1_test.txt")
	usetree = c45(trainingdata)
	testinglist = test(testingdata, usetree)
	for i in range(len(testinglist)):
		print(testinglist[i])


if __name__ == "__main__":
	main()

