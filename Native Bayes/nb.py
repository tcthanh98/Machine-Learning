from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pandas as pd

def load_data(number):
	if number == 1:
		training_set = pd.read_csv('data/fp/fp.trn', header=None)
		testing_set = pd.read_csv('data/fp/fp.tst', header=None)

	elif number == 2:
		training_set = pd.read_csv('data/iris/iris.trn', header=None)
		testing_set = pd.read_csv('data/iris/iris.tst', header=None)

	elif number == 3:
		training_set = pd.read_csv('data/letter/let.trn', header=None)			
		testing_set = pd.read_csv('data/letter/let.tst', header=None)

	elif number == 4:
		training_set = pd.read_csv('data/leukemia/ALLAML.trn', header=None)
		testing_set = pd.read_csv('data/leukemia/ALLAML.tst', header=None)

	elif number == 5:
		training_set = pd.read_csv('data/opt/opt.trn', header=None)
		testing_set = pd.read_csv('data/opt/opt.tst', header=None)

	else:	
		print("Dataset does not exist")
	return training_set, testing_set


def split_dataset(dataset):
	features = dataset.iloc[:, :-1]
	labels  = dataset.iloc[:, -1]
	return features, labels


def evaluate(y_test, y_predict):
	correct = 0

	for act, pred in zip(y_test, y_predict):
		if act == pred:
			correct += 1
	return correct/len(y_test)	

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


# def recall_precision_calc(matrix):
#     for i in range(len(matrix[0])):
#         row_values = matrix[i]
#         col_values = [row[i] for row in matrix]
#         tp = col_values[i]
#         fp = sum(row_values)-row_values[i]
#         fn = sum(col_values)-col_values[i]
    
#     recall = tp / (tp + fn)
#     precision = tp / (tp + fp)
    
#     F1_score = 2 * (precision * recall) / (precision + recall)
    
#     return recall, precision, F1_score



#load dataset:
print("\n1. fp\n2. iris\n3. letter\n4. leukemia\n5. opt\n-----------------------------------------------")
number = int(input("Please select dataset by number(1 - 5): "))
while (number != 1 and number != 2 and number != 3 and number != 4 and number != 5):
	number = int(input("Please select dataset by number(1 - 5): "))

training_set, testing_set = load_data(number)

# split dataset to features and labels
X_train, y_train = split_dataset(training_set)
X_test, y_test = split_dataset(testing_set)

#Create and train model
model = GaussianNB()
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#Evaluating model
# accuracy_score = evaluate(y_test, y_pred)
# print("\nGaussian Naive Bayes Accuracy (in %): ",accuracy_score)

# confuse matrix
accuracy_score = model.score(X_test, y_test)
print("\nGaussian Naive Bayes Accuracy (in %): ",accuracy_score* 100)

matrix = confusion_matrix(y_test, y_pred)
print('\nConfuse Matrix: ')
print(matrix)
print('\n')

