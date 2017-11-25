import numpy as np
import time
from math import exp

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

    totalEpoch = 5000
    weight = np.zeros(len(data[1,:]))
    j = 0

    for epoch in range(totalEpoch):

        for i in range(len(data)):

            j += 1
            alpha = 0.3/j
            pred = logisticFunction(weight, data[i,:])

            weight = weight + alpha*(target[i] - pred)*data[i,:]

        test(weight, data, target)

    return weight

def logisticFunction(weight, data):

    dotprod = weight.dot(data)

    value = 1.0 / (1 + exp(-dotprod))

    return value

def test(weight, data, target):

    errorCounter = 0.0

    for i in range(len(data)):

        prediction = logisticFunction(weight, data[i,:])

        if round(prediction) != target[i]:

            errorCounter += 1

    print "The error percent is {0:.2%}".format(errorCounter/len(data))

def standardizeData(data):

    mean = np.mean(data[:,:-1], axis=0)
    std = np.std(data[:,:-1], axis=0)

    data[:,:-1] = (data[:,:-1] - mean) / std

    return data
