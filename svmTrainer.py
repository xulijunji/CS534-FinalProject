import numpy as np
from sklearn import svm
import time
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def devPredictor(clf, devSet):

    #for i in enumerate(devSet):
    p = list(clf.predict(devSet))
    #print(p)

    return p

def SVM_quadratic(data, target, c = 1000, _kernel='poly', _degree=2, _coef0=1):

	data = data[:,2:]
	data[:,-1] = 1
	data = data.astype(float)

	clf = svm.SVC(kernel = _kernel, degree = _degree, C = c, coef0 = _coef0, probability=True)

	scores = cross_val_score(clf, data, target, cv=10)
	print("Accuracy for SVM_quadtratic: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def SVM_rbf(data, target):

        data = data[:,2:]
        data[:,-1] = 1
        data = data.astype(float)

        clf=svm.SVC(C=10000,gamma=0.025,kernel='rbf')
	#clf.fit(data, target)

	#train_error = clf.score(data, target)
	#trainArr.append((1-train_error) * 100)

        scores = cross_val_score(clf, data, target, cv=10)
	#devArr.append(100-scores.mean())
        print("Accuracy for SVM rbf: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	

def knn_fit(data, target):

    data = data[:,2:]
    data[:,-1] = 1
    data = data.astype(float)

    for i in range(1, 57, 5):

            neigh = KNeighborsClassifier(n_neighbors=i)
	    #neigh.fit(data, target)
    
	    #train_error = neigh.score(data, target)
	    #print("train_error: ", train_error)
    	    #print(neigh.predict_proba(data))

            scores = cross_val_score(neigh, data, target, cv=30)
	    #devArr.append(100-scores.mean())
            print("i: %d", i)
            print()
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	

    #kf = KFold(n_splits=2)

    #for data, target in kf.split(X):
    #    print("%s %s" % (train, test))

def SVM_fit1(data, target, c = 1, _kernel='linear', _degree=1, _coef0=0):
    
    x = np.delete(data, 1, 1)
    #print("len: ", len(x[0]))

    print(type(x[0][3]))

    #x = np.delete(x, len(x[0]) - 1, 1)
    #print(x)

    data = data[:,2:]
    data[:,-1] = 1
    data = data.astype(float)

    print(data)

    trainArr = []
    devArr = []
    cArr = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000]

    clf = svm.SVC(kernel = _kernel, degree = _degree, C = 100, coef0 = _coef0, probability=True)

    y_pred_svc_p =clf.predict_proba(data)[:,1]

    models=[y_pred_svc_p]
    label=['SVM']

    # plotting ROC curves
    plt.figure(figsize=(10, 8))
    m=np.arange(1)
    for m in m:
    	fpr, tpr,thresholds= metrics.roc_curve(target,models[m])
    	print('model:',label[m])
    	print('thresholds:',np.round(thresholds,3))
    	print('tpr:       ',np.round(tpr,3))
    	print('fpr:       ',np.round(fpr,3))
    	plt.plot(fpr,tpr,label=label[m])
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.title('ROC curve for Cancer classifer')
    plt.xlabel('False positive rate (1-specificity)')
    plt.ylabel('True positive rate (sensitivity)')
    plt.legend(loc=4)
    plt.show()


def SVM_fit10(data, target, c = 1, _kernel='linear', _degree=1, _coef0=0):
    
    x = np.delete(data, 1, 1)
    #print("len: ", len(x[0]))

    print(type(x[0][3]))

    #x = np.delete(x, len(x[0]) - 1, 1)
    #print(x)

    data = data[:,2:]
    data[:,-1] = 1
    data = data.astype(float)

    print(data)

    trainArr = []
    devArr = []
    cArr = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

    clf = svm.SVC(kernel = _kernel, degree = _degree, C = 100, coef0 = _coef0, probability=True)

    startTime = time.time()

        #currTarget = np.concatenate((target[:i], target[i+10:]), axis = 0).tolist()
        #clf.fit(np.concatenate((data[:i], data[i+10:]), axis = 0), np.concatenate((target[:i], target[i+10:]), axis = 0))

    clf.fit(data, target)
    endTime = time.time()

    train_error = clf.score(data, target)
    trainArr.append((1-train_error) * 100)
        #print("For C: ", v)
        #print("Train Error: ", (1-train_error) * 100)

        #print("The SVM ran for %s seconds: " % (endTime - startTime))
        #print("Number of Support Vectors: ", str(len(clf.support_vectors_)))

    scores = cross_val_score(clf, data, target, cv=10)
    devArr.append(100-scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	
        #print("Dev Error rate:", devArr[i])

    weightVector = clf.coef_
    max_index = weightVector.argsort()[-3:][::-1]

    print("max weights: ", max_index)
        #print()

    print("predicted: ", clf.predict(data))

    plt.plot(cArr, trainArr, label='C vs Train Error')
    plt.plot(cArr, devArr, label='C vs Dev Error')
    plt.title('Error Rates')
    plt.legend()
    plt.xlabel('C')
    plt.ylabel('Error rates')


def SVM_fit(data, target, c = 1, _kernel='linear', _degree=1, _coef0=0):
    
    x = np.delete(data, 1, 1)
    #print("len: ", len(x[0]))

    print(type(x[0][3]))

    #x = np.delete(x, len(x[0]) - 1, 1)
    #print(x)

    data = data[:,2:]
    data[:,-1] = 1
    data = data.astype(float)

    print(data)

    trainArr = []
    devArr = []
    cArr = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

    for i, v in enumerate(cArr, 0):

        clf = svm.SVC(kernel = _kernel, degree = _degree, C = v, coef0 = _coef0, probability=True)

        startTime = time.time()

        #currTarget = np.concatenate((target[:i], target[i+10:]), axis = 0).tolist()
        #clf.fit(np.concatenate((data[:i], data[i+10:]), axis = 0), np.concatenate((target[:i], target[i+10:]), axis = 0))

        data_norm = (data - data.mean()) / (data.std()) 
        clf.fit(data_norm, target)
        endTime = time.time()

        train_error = clf.score(data_norm, target)
        trainArr.append((1-train_error) * 100)
        #print("For C: ", v)
        #print("Train Error: ", (1-train_error) * 100)

        #print("The SVM ran for %s seconds: " % (endTime - startTime))
        #print("Number of Support Vectors: ", str(len(clf.support_vectors_)))

        scores = cross_val_score(clf, data_norm, target, cv=10)
        devArr.append((1-scores.mean())*100)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	
        #print("Dev Error rate:", devArr[i])

        weightVector = clf.coef_
        max_index = weightVector.argsort()[-3:][::-1]

        print("max weights: ", max_index)
        #print()

    #print("predicted: ", clf.predict(data))

    plt.plot(cArr, trainArr, label='C vs Train Error')
    plt.plot(cArr, devArr, label='C vs Dev Error')
    plt.title('Error Rates')
    plt.legend()
    plt.xlabel('C')
    plt.ylabel('Error rates')

    plt.show()

    return clf

def SVM_fit2(data, target):

        data = data[:,2:]
        data[:,-1] = 1
        data = data.astype(float)
        svc=SVC(C=100,gamma=0.001,kernel='rbf')
        svc.fit(data, target)

	# for display purposes, we fit the model on the first two components i.e. PC1, and PC2
        svc.fit(data[:,0:2], target)

	# Plotting the decision boundary for all data (both train and test)
	# Create color maps
	
        cmap_light = ListedColormap(['#AAFFAA','#FFAAAA'])
        cmap_bold = ListedColormap(['#0000FF','#FF0000'])

	# creating a meshgrid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h=0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        xy_mesh=np.c_[xx.ravel(), yy.ravel()]
        Z = svc.predict(xy_mesh)
        Z = Z.reshape(xx.shape)

	#plotting data on decision boundary
	
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('PC1');plt.ylabel('PC2')
        plt.title('SVC')
        plt.show()
