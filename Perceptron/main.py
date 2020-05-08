import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from perceptron import Perceptron

def load_data(number):
	if number == 1:
		training_set = pd.read_csv('data/leukemia/ALLAML.trn', header=None)
		testing_set = pd.read_csv('data/leukemia/ALLAML.tst', header=None)
	elif number == 2:
		dataset = pd.read_csv('data/spam/spam.data.')
		dataset_copy = dataset.copy()
		training_set = dataset_copy.sample(frac=0.7, random_state=0)
		testing_set = dataset_copy.drop(training_set.index)

	elif number == 3:
		dataset = pd.read_csv('data/ovarian/ovarian.data.')
		dataset_copy = dataset.copy()
		training_set = dataset_copy.sample(frac=0.7, random_state=0)
		testing_set = dataset_copy.drop(training_set.index)

	else:	
		print("Dataset does not exist")

	return training_set, testing_set
		
def split_dataset(dataset):
	features = dataset.iloc[:, :-1].values
	labels  = dataset.iloc[:, -1].values
	return features, labels


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy	

# def confusion_matrix(actual, predicted):
# 	unique = set([row[-1] for row in actual])
# 	matrix = [list() for x in range(len(unique))]
# 	for i in range(len(unique)):
# 		matrix[i] = [0 for x in range(len(unique))]
# 	lookup = dict()
# 	for i, value in enumerate(unique):
# 		lookup[value] = i
# 	for i in range(len(actual)):
# 		x = lookup[actual[i][-1]]
# 		y = lookup[predicted[i]]
# 		matrix[x][y] += 1
# 	return unique, matrix

# def print_confusion_matrix(unique, matrix):
# 	print('Unique prediction values:')
# 	print('(P)' + ' '.join(str(x) for x in unique) + '\n')

# 	print("Confusion Matrix:")
# 	for i, x in enumerate(unique):
# 		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))		


print("\n1. Spam\n2. Leukemia\n3. Ovarian\n-----------------------------------------------")

number = int(input("Please select dataset by number(1 - 3): "))
while (number != 1 and number != 2 and number != 3 and number != 4 and number != 5):
	number = int(input("Please select dataset by number(1 - 3): "))


#load data
training_set, testing_set = load_data(number)

# split dataset to features and labels
X_train, y_train = split_dataset(training_set)
X_test, y_test = split_dataset(testing_set)


p = Perceptron(learning_rate=0.01, n_iters=100)
p.fit(X_train, y_train)
pred_list = p.predict(X_test)

print("\nAccuracy", accuracy(y_test, pred_list))

df_confusion = pd.crosstab(y_test, pred_list, rownames=['Actual '], colnames=['Predict'])
print('\n',df_confusion)