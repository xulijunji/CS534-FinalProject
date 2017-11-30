import numpy as np
from sklearn import svm
import time
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

def gradient_booster(data, target):

	data = data[:,2:]
	data[:,-1] = 1
	data = data.astype(float)

	for i in range(150, 400, 50):

		model= GradientBoostingClassifier(n_estimators=i, learning_rate=0.5, max_depth=2, random_state=0)
		model.fit(data, target)

		train_error = model.score(data, target)
		print("train_Error: ", train_error)

		scores = cross_val_score(model, data, target, cv=5)
	    	#devArr.append(100-scores.mean())
		print("Accuracy: %0.2f (+/- %0.2f) for n_estimators %d " % (scores.mean(), scores.std() * 2, i))	

	#Predict Output
	#predicted= model.predict(x_test)
