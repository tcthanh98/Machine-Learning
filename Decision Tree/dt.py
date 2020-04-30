from sklearn.tree import DecisionTreeClassifier
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


#load dataset:
print("\n1. fp\n2. iris\n3. letter\n4. leukemia\n5. opt\n-----------------------------------------------")
number = int(input("Please select dataset by number(1 - 5): "))
while (number != 1 and number != 2 and number != 3 and number != 4 and number != 5):
	number = int(input("Please select dataset by number(1 - 5): "))

training_set, testing_set = load_data(number)

# split dataset to features and labels
X_train, y_train = split_dataset(training_set)
X_test, y_test = split_dataset(testing_set)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score = model.score(X_test, y_test)
print("\nDecision Tree Classifier Accuracy (in %): ",accuracy_score* 100)

matrix = confusion_matrix(y_test, y_pred)
print('\nConfuse Matrix: ')
print(matrix)
print('\n')


