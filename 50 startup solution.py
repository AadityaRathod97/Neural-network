# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:08:17 2020

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Build a neural network model for predicting profits of the startup.
'''
startup = pd.read_csv("50_Startups.csv")
startup.columns
plt.hist(startup.Profit)

startup.loc[startup.Profit < 105000,"Profit"] = 1 #profit as low
startup.loc[startup.Profit > 105000,"Profit"] = 2 #profit as high

startup.Profit.value_counts()

#to convert string fields to numeric
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
startup["State"] = number.fit_transform(startup["State"])

X = startup.drop(["Profit"],axis=1)
Y = startup["Profit"]
plt.hist(Y)
startup.Profit.value_counts()

startup.corr()
#spliting the data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y)

#applying scale to the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
#creating neural network model
mlp = MLPClassifier(hidden_layer_sizes=(25,25))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_train==prediction_train) #1
np.mean(y_test==prediction_test) #1
