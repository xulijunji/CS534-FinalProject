import numpy as np
from sklearn import svm
import time

def SVM_fit(data, target, c = 1, _kernel='linear', _degree=1, _coef0=0):
    
    x = np.delete(data, 1, 1)
    #print("len: ", len(x[0]))

    x = np.delete(x, len(x[0]) - 1, 1)
    print(x)

    clf = svm.SVC(kernel = _kernel, degree = _degree, C = c, coef0 = _coef0)

    startTime = time.time()
    clf.fit(x, target)
    endTime = time.time()

    print("The SVM ran for %s seconds: " % (endTime - startTime))
    print("Number of Support Vectors: ", str(len(clf.support_vectors_)))

    return clf
