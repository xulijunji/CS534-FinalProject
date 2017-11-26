#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import svm
import time
from svmTrainer import SVM_fit
from logisticTrainer import logisticRegression

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

        _kernel = int(input("Kernel [1: linear | 2: quadratic] > "))
        kernel = 'linear'
        degree = 1
        coef0 = 0

        if _kernel == 2:
            kernel = 'poly'
            degree = 2
            coef0 = 1

        cParam = float(input("c Parameter > "))

        clf = SVM_fit(npData, newTarget, cParam, kernel, degree, coef0)


    if algorithm == 2:

        model = logisticRegression(npData, newTarget)
