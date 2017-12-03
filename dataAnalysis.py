import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analysis():

	data = pd.read_csv('data.csv')
	
	y = data.diagnosis	# M or B
	list = ['Unnamed: 32','id','diagnosis']
	x = data.drop(list, axis = 1)
	
	ax = sns.countplot(y, label="count")
	B, M = y.value_counts()
	print('Number of Benign: ',B)
	print('Number of Malignant : ',M)

	plt.show()

	data_dia = y
	data = x
	data_n_2 = (data - data.mean()) / (data.std())              # standardization
	data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
	data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
	plt.figure(figsize=(10,10))
	sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
	plt.xticks(rotation=90)

	plt.show()

if __name__ == "__main__":

	analysis()
