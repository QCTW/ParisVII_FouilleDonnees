from sklearn.cluster import KMeans

def RunKMeans(X, n_clusters):
	model = KMeans(n_clusters = n_clusters)
	model.fit(X) # X the result of TF-IDF
	# Groupe attribute au i-eme message
	return model.labels_

#Silhoutte_score
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5) #To split the data in 2 for test and training

from sklearn.model_selection import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(X_train, y_train)
predictions = model.predict(X_test) #Array that predicts if each element in the array is a spam or not
predictions_p = model.predit_proba(X_test) # To see the probability of each elements to be 1 or 0

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Precision pour la classe 0: P(y=0| y'=0) ==> For all the spam we mark, the percentage of which we are correct
# Recall: P(y'=0|y=0) ===> For all the spam, we did mark it as spam

k_fold = StratifiedKFold(n_split=10)
k_fold.get_n_splits(X_train, y_train)
accuracy = np.zeros((10, 50))
for index_train, index_test in k_fold.split(X_train, y_train):
	X_train_fold = X_train[index, :]
	X_test_fold = X_train[index_test, :]
	y_train_fold = y_train[index_train]
	y_test_fold = y_train[index_test]
	for n_neighbors in np.arrange(1, 100, 2):
		model = KNeighborsClaasifier(n_neighbors = n_neighbors)
		model.fit(X_train_fold, y_train_fold)
		predictions = model.predict(X_test_fold)
		accuracy[fold, n_neighbors//2] = accuracy_score(y_test_fold, predictions)


# Precision : P (y=1 | yPredict=1) 
# Recall    : P (yPredict=1 | y=1) 
# Ex :
# PredictY = {0 1 1 0 0 1 0}
# RealY    = {0 0 1 0 1 0 0}
# Precision = P(y=1|yPredict=1) = 1 / 3 ---> Three 1 in our prediction and One of our predict is correct
# Recall    = P(yPredict=1| y=1) = 1 / 2 ---> Two 1 in RealY and we predict One correct (at index 2)

def train_test_model():
model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train, y_train)
final_predictions = model.predict(X_test)
final_accuracy = accuracy_score(final_predictions, y_test)
final_recall = recall_score(final_predictions, y_test)
print(final_accuracy)
print(final_recall)

