import numpy as np
import time
from math import exp
from sklearn.model_selection import KFold

def logisticRegression(data, rawTarget):

    data = data[:,2:]
    data[:,-1] = 1
    data = data.astype(float)
    data = standardizeData(data)
    target = []

    for value in rawTarget:

        if value == -1:

            target.append(0)

        else:

            target.append(1)

    target = np.asarray(target)
    k = 10
    kf = KFold(n_splits=k)
    n = 1
    scoreList = []
    cacheWeight = np.zeros(len(data[1,:]))

    for train_index, test_index in kf.split(data):

        print "Beginning cross-fold validation {}.".format(str(n))

        trainData = data[train_index]
        trainTarget = target[train_index]
        devData = data[test_index]
        devTarget = target[test_index]

        totalEpoch = 500
        weight = np.zeros(len(data[1,:]))
        j = 0

        for epoch in range(totalEpoch):

            for i in range(len(data)):

                j += 1
                alpha = 0.1/j
                pred = logisticFunction(weight, data[i,:])

                weight = weight + alpha*(target[i] - pred)*data[i,:]

        _ = test(weight, trainData, trainTarget, 'training')
        score = test(weight, devData, devTarget, 'dev')
        scoreList.append(score)
        cacheWeight += weight
        n += 1

    scoreList = np.asarray(scoreList)
    print "Average Error Rate: {:.2%} (+/- {:.2%})".format(scoreList.mean(), scoreList.std()*1.96)
    _ = test(cacheWeight/k, data, target, 'training')
    prediction = predict(cacheWeight/k, data, target)

    print(cacheWeight/k)

    return cacheWeight/k, prediction

def logisticFunction(weight, data):

    dotprod = weight.dot(data)

    value = 1.0 / (1 + exp(-dotprod))

    return value

def test(weight, data, target, trainType):

    errorCounter = 0.0

    for i in range(len(data)):

        prediction = logisticFunction(weight, data[i,:])

        if round(prediction) != target[i]:

            errorCounter += 1

    print "The {} error rate is {:.2%}".format(trainType, errorCounter/len(data))

    return errorCounter/len(data)

def predict(weight, data, target):

    predictionList = []

    for i in range(len(data)):

        prediction = logisticFunction(weight, data[i,:])
        predictionList.append(prediction)

    return predictionList

def standardizeData(data):

    mean = np.mean(data[:,:-1], axis=0)
    std = np.std(data[:,:-1], axis=0)

    data[:,:-1] = (data[:,:-1] - mean) / std

    return data
