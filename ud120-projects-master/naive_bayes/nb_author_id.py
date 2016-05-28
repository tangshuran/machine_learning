#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append(r"E:\machine_learning\ud120-projects-master\tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
t0 = time()
clf = GaussianNB()#TODO
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
### fit the classifier on the training features and labels
#TODO

### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)#TODO
print "predictin time:", round(time()-t1, 3), "s"


### calculate and return the accuracy on the test data
### this is slightly different than the example, 
### where we just print the accuracy
### you might need to import an sklearn module
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)#TODO

#########################################################


