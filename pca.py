import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def principalComponentAnalysis(data, target):

    target = np.asarray(target)
    data = data[:,1:]
    pca = PCA(n_components=2)
    data_r = pca.fit(data).transform(data)
    
    plt.figure()
    colors = ['navy', 'darkorange']
    for color, i, target_name in zip(colors, [0,1], ['benign', 'malignant']):
        plt.scatter(data_r[target == i, 0], data_r[target == i, 1], color=color, alpha=.8, lw=2, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Principal Component Analysis')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    for i, target_name in zip([0,1], ['benign', 'malignant']):
        plt.hist(data[target==i, 26],
                 label=target_name,
                 bins=10,
                 alpha=0.3,)
    plt.xlabel('Concavity Worst')

    plt.subplot(1, 2, 2)
    for i, target_name in zip([0,1], ['benign', 'malignant']):
        plt.hist(data[target==i, 20],
                 label=target_name,
                 bins=10,
                 alpha=0.3,)
    plt.xlabel('Radius Worst')
    
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

    plt.tight_layout()
    plt.show()

def kMeansClustering(data, target):

    target = np.asarray(target)
    data = data[:,2:]
    model = KMeans(n_clusters=2)
    model.fit(data)

    # Plot the Original Classifications
    plt.subplot(1, 2, 1)
    plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[target], s=40)
    plt.title('Real Classification')

    # Plot the Models Classifications
    plt.subplot(1, 2, 2)
    plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
    plt.title('K Mean Classification')

    
    print(target)
    print(model.labels_ )
