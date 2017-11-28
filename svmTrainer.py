import numpy as np
from sklearn import svm
import time
from sklearn.model_selection import cross_val_score

def devPredictor(clf, devSet):

    #for i in enumerate(devSet):
    p = list(clf.predict(devSet))
    #print(p)

    return p

def SVM_fit(data, target, c = 1, _kernel='linear', _degree=1, _coef0=0):
    
    x = np.delete(data, 1, 1)
    #print("len: ", len(x[0]))

    print(type(x[0][3]))

    #x = np.delete(x, len(x[0]) - 1, 1)
    #print(x)

    data = data[:,2:]
    data[:,-1] = 1
    data = data.astype(float)

    clf = svm.SVC(kernel = _kernel, degree = _degree, C = c, coef0 = _coef0)

    j = 0
    errors = 0

    #for i in range(0, len(data)-10, 10):

    startTime = time.time()
            #currTarget = np.concatenate((target[:i], target[i+10:]), axis = 0).tolist()
    #clf.fit(np.concatenate((data[:i], data[i+10:]), axis = 0), np.concatenate((target[:i], target[i+10:]), axis = 0))
    clf.fit(data, target)
    endTime = time.time()
            #l = devPredictor(clf, data[i:i+10])
            #npl = np.asarray(l)
            #npl = npl.astype(int)
            #print("devSet :", i , i+10, "predicted: ", l )
            #print(type(npl), npl)
            #errors += sum(t1!=t2 for (t1,t2) in zip(l, currTarget))
            #print(errors)

    train_error = clf.score(data, target)
           # dev_error = clf.score(data[i:i+10], target[i:i+10])

    #print("Dev Error rate:", (1-dev_error) * 100)

    print("Train Error: ", (1-train_error) * 100)
    #print("Dev Error rate:", (1-dev_error) * 100)

    print("The SVM ran for %s seconds: " % (endTime - startTime))
    print("Number of Support Vectors: ", str(len(clf.support_vectors_)))

    scores = cross_val_score(clf, data, target, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf
