#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

t0 = time()
clf = svm.SVC(kernel='rbf',C=10000.)#TODO
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
clf.fit(features_train,labels_train)
#print "training time:", round(time()-t0, 3), "s"
### fit the classifier on the training features and labels
#TODO

### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)#TODO
#print "predictin time:", round(time()-t1, 3), "s"


### calculate and return the accuracy on the test data
### this is slightly different than the example, 
### where we just print the accuracy
### you might need to import an sklearn module
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)#TODO
print accuracy



#########################################################
### your code goes here ###

#########################################################


