# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:46:12 2021

@author: User
"""

#preprocess here in python


#read data file into a numpy array
import numpy
from numpy import genfromtxt
trainingData = genfromtxt("applications_train.csv", delimiter = ",") ## your data has to be only numbers
testingData = genfromtxt("applications_test.csv", delimiter = ",")


#consider training and testing
#1. training and testing are separated
#testdata = genfromtxt("covertype12-10percent_test.csv", delimiter = ",")

#2. training and testing are in one file
#pick first 70% of data to be training
#the last 30% of data is testing
# train = trainingData[0:int(0.7*len(trainingData)), : ];
# test = trainingData[int(0.7*len(trainingData)):len(trainingData), : ];


#build model using training data

from sklearn.linear_model import LogisticRegression

trainX = trainingData[ : , 0 : 5]
trainY = trainingData[ : , 54]

clf = LogisticRegression(random_state=0,solver="lbfgs").fit(trainX, trainY)


#test the model performance
testX = testingData[:,0:5]
testY = testingData[:,54]
results = clf.predict(testX)


from sklearn.metrics import accuracy_score
accu = accuracy_score(testY, results)

print(accu)


#cross validation
from sklearn.model_selection import cross_val_score
clf2 = LogisticRegression(random_state=0,solver="lbfgs")
scores = cross_val_score(clf, trainingData[:,0:5], trainingData[:,54], cv=5)
print(scores)