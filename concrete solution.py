# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:14:05 2020

@author: DELL
"""


'''
Prepare a model for strength of concrete data using Neural Networks
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


concrete = pd.read_csv("concrete.csv")
concrete.columns
plt.hist(concrete.strength)

concrete.loc[concrete.strength < 30,"strength"] =  0 #weak
concrete.loc[concrete.strength >= 30,"strength"] =  1 #strength

X = concrete.iloc[:,0:8]
Y = concrete.iloc[:,8]

#spliting the data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y)

from sklearn import preprocessing
# standardize the data attributes
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

from sklearn.neural_network import MLPClassifier
#creating neural network model
mlp = MLPClassifier(hidden_layer_sizes=(100,100))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_train==prediction_train) #0.97
np.mean(y_test==prediction_test) #0.90
