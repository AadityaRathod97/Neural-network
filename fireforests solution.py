# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:11:29 2020

@author: DELL
"""


'''
PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


forestfires = pd.read_csv("forestfires.csv")
forestfires.columns
plt.hist(forestfires.size_category)

forestfires.size_category.value_counts()

#converting the category field in numeric
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
forestfires["size_category"] = number.fit_transform(forestfires["size_category"])

#dropping the unnecessary columns
forestfires.drop("month",axis=1,inplace=True)
forestfires.drop("day",axis=1,inplace=True)

X = forestfires.iloc[:,0:28]
Y = forestfires.iloc[:,28]

#spliting the data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y)

from sklearn import preprocessing
# standardize the data attributes
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

from sklearn.neural_network import MLPClassifier
#creating neural network model
mlp = MLPClassifier(hidden_layer_sizes=(500,500))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_train==prediction_train) #1
np.mean(y_test==prediction_test) #0.90

