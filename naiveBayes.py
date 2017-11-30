import numpy as np
from sklearn import svm
import time
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def naiveBayes(data, target):

	data = data[:,2:]
	data[:,-1] = 1
	data = data.astype(float)

	#gnb = GaussianNB()
	gnb = MultinomialNB(alpha = 1)
	#gnb = BernoulliNB()
	gnb.fit(data, target)

	train_error = gnb.score(data, target)
	print("train_Error: ", train_error)

	scores = cross_val_score(gnb, data, target, cv=10)
    	#devArr.append(100-scores.mean())
	print("Accuracy: %0.2f (+/- %0.2f) for n_estimators " % (scores.mean(), scores.std() * 2))	

