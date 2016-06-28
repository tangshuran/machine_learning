#!/usr/bin/python

import sys
import pickle
import os
import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt

os.chdir(r"E:\machine_learning\ud120-projects-master\final_project")
sys.path.append(r"E:\machine_learning\ud120-projects-master/tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list =  ['salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
def summarise_data(feature,data=data_dict):
    num_of_nans=0
    temp=[]
    for people,info in data.iteritems():
        if info[feature]=="NaN":
            num_of_nans +=1
        else:
            temp.append(info[feature])
    result=describe(temp)
    print "num_of_nans:",num_of_nans
    print ""
    print "describe the feature:",result
        
    
    
### Task 2: Remove outliers
#find the existance of the outlier through visualization
def pre_visualization(column1,column2,data=data_dict):
    x=[]
    y=[]
    for people,info in data.iteritems():
        if info[column1]!="NaN" and info[column2]!="NaN":
            x.append(info[column1])
            y.append(info[column2])
    return x,y
x,y=pre_visualization("salary",'bonus')
plt.plot(x,y,".")
plt.show() #we can see that there is a point at the top right part of the plot, that is a outlier
#next thing to do is to find the outlier, and delete it
def get_a_feature(feature,data=data_dict):
    value=[]
    name=[]
    for people,info in data.iteritems():
        if info[feature]!="NaN":
            value.append(info[feature])
            name.append(people)
    return zip(name,value)
salary=get_a_feature("salary")
sorted_salary=sorted(salary, key=lambda x:x[1],reverse=True)
#Then we can find the bad guy
print sorted_salary[0]
#delete the ourlier from our data
data_dict.pop(sorted_salary[0][0])

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#create a new feature named "stock_to_salary_ratio"
my_dataset = data_dict
for each in my_dataset.values():
    if each['salary']!="NaN" and each['total_stock_value'] !="NaN":
        each["stock_to_salary_ratio"]=float(each['total_stock_value'])/each['salary']
    else:
        each["stock_to_salary_ratio"]="NaN"
features_list.extend(["stock_to_salary_ratio"])
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
