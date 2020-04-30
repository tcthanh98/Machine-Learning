import pandas as pd
import numpy as np
from math import sqrt

def load_data(filename):
	dataset = list()
	with open(filename, 'r') as file:
		for row in file:
			if not row:
				continue
			dataset.append(row.rstrip('\n').split(','))
	return dataset


def Euclidean_distance(instance1, instance2):
	distance = 0
	dimension = len(instance1) - 1
	for i in range(dimension):
		distance += (float(instance1[i]) - float(instance2[i]))**2
	return sqrt(distance)


def predict(k, train_set, test_instance):
	distance_list = []

	for i in range(len(train_set)):
		dist = Euclidean_distance(train_set[i][:-1], test_instance )
		distance_list.append((train_set[i], dist))

	distance_list.sort(key=lambda x: x[1])	

	neighbor_list = []
	for i in range(k):
		neighbor_list.append(distance_list[i][0])

	classes = {}
	for i in range(len(neighbor_list)):
		response = neighbor_list[i][-1]
		if response in classes:
			classes[response] += 1
		else:
			classes[response] = 1
	
	sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
	
	return sorted_classes[0][0]


def evaluate(label_true, label_predict):
	correct = 0

	for act, pred in zip(label_true, label_predict):
		if act == pred:
			correct += 1
	return correct/len(label_true)		

# Calculate a Confusion Matrix #	
def confusion_matrix(actual, predicted):
	unique = set([row[-1] for row in actual])
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i][-1]]
		y = lookup[predicted[i]]
		matrix[x][y] += 1
	return unique, matrix

# Printing a confusion matrix
def print_confusion_matrix(unique, matrix):
	print('Unique prediction values:')
	print('(P)' + ' '.join(str(x) for x in unique) + '\n')

	print("Confusion Matrix:")
	for i, x in enumerate(unique):
		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))


def recall_precision_calc(matrix):
    for i in range(len(matrix[0])):
        row_values = matrix[i] # row values of matrix
        col_values = [row[i] for row in matrix] # column values of matrix
        tp = col_values[i]
        fp = sum(row_values)-row_values[i] # sum all row values - ones in diagonal
        fn = sum(col_values)-col_values[i] # sum all col values - ones in diagonal
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    
    F1_score = 2 * (precision * recall) / (precision + recall)
    
    return recall, precision, F1_score

# load dataset
train_set = load_data("data/iris/iris.trn")
test_set = load_data("data/iris/iris.tst")


# # Find the optimal value for k

# k_evaluations = []

# actual = np.array(test_set)[:, -1]

# for k in range(1, 22, 2):
# 	pred_list = []
# 	for row in test_set:
# 		predictors_only = row[:-1]
# 		prediction = predict(k, train_set, predictors_only)
# 		pred_list.append(prediction)

# 	current_accuracy = evaluate(actual, pred_list)
# 	k_evaluations.append((k, current_accuracy))
# print(k_evaluations)	



pred_list = []
for row in test_set:
	predictors_only = row[:-1]
	prediction = predict(3, train_set, predictors_only)
	pred_list.append(prediction)

actual = np.array(test_set)[:, -1]

accuracy = evaluate(actual, pred_list)
print("Accuracy = " + str(accuracy) + '%')

# confuse matrix
unique, matrix = confusion_matrix(test_set, pred_list)
print('\n')
print_confusion_matrix(unique, matrix)
print('\n')

Recall, Precision, F1_score = recall_precision_calc(matrix)
print('Recall:', Recall)
print('Precision:', Precision)
print('F1 score:', F1_score)

print(test_set)
print(pred_list)