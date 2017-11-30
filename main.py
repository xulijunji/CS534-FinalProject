#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import svm
import time
from svmTrainer import knn_fit
from svmTrainer import SVM_fit
from svmTrainer import SVM_fit1
from svmTrainer import SVM_fit10
from logisticTrainer import logisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from gradientbooster import gradient_booster

datafile = "data.csv"

if __name__ == "__main__":

    data = pd.read_csv(datafile)

    npData = data.values

    target = npData[:, 1]
    newTarget = []

    for i, v in enumerate(target, 0):

        if v == 'M':
            y = 1

        else:
            y = -1

        newTarget.append(y)
    

    algorithm = int(input("Select what you want to do with the dataset > "))

    if algorithm == 0:
        print(0)
        #insert your algorithm here and pass data as argument
        
    if algorithm == 1:

        train_file = "data.csv"
        cArr = []
        supportVectors = []

        #knn_fit(npData, newTarget)

	m = gradient_booster(npData, newTarget)

        while(1):

               _kernel = int(input("Kernel [1: linear | 2: quadratic] > "))
       	       kernel = 'linear'
               degree = 1
               coef0 = 0

               if _kernel == 2:
                    kernel = 'poly'
                    degree = 2
                    coef0 = 1

               cParam = float(input("c Parameter > "))

               #clf = SVM_fit(npData, newTarget, cParam, kernel, degree, coef0)

	       

               #clf = svm.SVC(kernel = 'rbf', C = 100, gamma = '0.25', probability=True)
	       #scores = cross_val_score(clf, data, newTarget, cv=10)
	       #devArr.append(100-scores.mean())

        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
  #             cArr.append(cParam)
   #            supportVectors.append(len(clf.support_vectors_))

                #plt.plot(cArr, supportVectors, label='C vs Support Vectors')
            #plt.plot(epochs, devArr, label='Dev Error')
        #plt.title('Error Rates for Unaveraged and Averaged Perceptron')
        #plt.legend()
        #plt.xlabel('C')
        #plt.ylabel('Support Vectors')
        #plt.show()


    if algorithm == 2:

        model, prediction = logisticRegression(npData, newTarget)
