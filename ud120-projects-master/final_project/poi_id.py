#!/usr/bin/python

import sys
import pickle
from scipy.stats import describe
import matplotlib.pyplot as plt
import pprint as pp

sys.path.append(r"../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list =  [
'poi',
'salary', 
'exercised_stock_options',
'bonus', 
'total_stock_value',
#"deferred_income", 
#'expenses', 
#'total_payments', 
#'long_term_incentive',
#'restricted_stock', 
#'to_messages',
#'from_poi_to_this_person', 
#'from_messages', 
#'from_this_person_to_poi',
#'shared_receipt_with_poi'
] # You will need to use more features

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
    print "describe the feature:",result
        
poi_number=0
non_poi_number=0
features_number=len(data_dict.items()[0][1])
for a in data_dict:
    if data_dict[a]["poi"]== True:
        poi_number +=1
    elif data_dict[a]["poi"]== False:
        non_poi_number +=1
    else:
        pass
print "number of poi is:",poi_number
print "number of non poi is:", non_poi_number
print "features_number is:", features_number    
    
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
print "The outlier:"
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
#features_list.extend(["stock_to_salary_ratio"])
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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#scaler = StandardScaler()
#kBest = SelectKBest(f_classif)
#gnb = GaussianNB()
#
#pipeline = Pipeline([#('fscale', scaler),
#                     ('fselect', kBest),
#                     ('gnb', gnb)])
#params_test = dict(fselect__k = [2, 3, 4])
############################################################
scaler = StandardScaler()
kBest = SelectKBest()
knn = KNeighborsClassifier()

pipeline = Pipeline([#('fscale', scaler), 
                     ('fselect', kBest),
                     ('knn', knn)])

params_test = dict(fselect__score_func = [f_classif],
                  knn__n_neighbors=[3, 5, 7],
                  fselect__k = [2, 3, 4],
                  knn__weights = ['distance', 'uniform'],
                  knn__algorithm = ['brute', 'kd_tree', 'ball_tree'],
                  knn__leaf_size = [1, 3, 5])
###############################################################
#kBest = SelectKBest(f_classif)
#tree = DecisionTreeClassifier()
#
#pipeline = Pipeline([('fselect', kBest),
#                     ('tree', tree)])
#
#params_test = dict(fselect__k = [2, 3, 4],
#                   tree__min_samples_split = [2, 3, 4, 5],
#                   tree__criterion = ['gini', 'entropy'],
#                   tree__max_features = [1, 2])
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.3, random_state=42)
#################################################
#without StratifiedShuffleSplit
#################################################
#with StratifiedShuffleSplit
n_iters = 100
r_state = 42
cv = StratifiedShuffleSplit(labels, n_iter = n_iters, random_state = r_state)
grid_search = GridSearchCV(pipeline, params_test, cv = cv, scoring = "f1")
grid_search.fit(features, labels)
best_params=grid_search.best_params_
print "the best parameters for this clf is:"
pp.pprint(best_params)
clf = pipeline.set_params(**best_params)
print "Scores of the features:"
print pipeline.named_steps['fselect'].fit(features, labels).scores_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#clf.fit(features_train,labels_train)
#pred=clf.predict(features_test)
#precision=precision_score(labels_test,pred)
#recall=recall_score(labels_test,pred)
#print "precision :",precision
#print "recall :",recall
pred_class = []
actual_class = []

# Calculate precision and recall and report on evaluation metrics
#cited from https://github.com/Maanum/DAND-P5-IdentifyEnronFraud
for train_indices, test_indices in cv:
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]

    clf.fit(features_train, labels_train)
    pred_class.extend(clf.predict(features_test))
    actual_class.extend(labels_test)
print
print 'Results ({} iterations, random state {}):'.format(n_iters, r_state)
print "\tPrecision: ", precision_score(actual_class, pred_class)
print "\tRecall:", recall_score(actual_class, pred_class)
dump_classifier_and_data(clf, my_dataset, features_list)
