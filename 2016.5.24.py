#import numpy as np
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(X, Y)

#from prep_terrain_data import makeTerrainData
#from class_vis import prettyPicture, output_image
#from ClassifyNB import classify
#
#import numpy as np
#import pylab as pl
#
#
#features_train, labels_train, features_test, labels_test = makeTerrainData()
#
#### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
#### in together--separate them so we can give them different colors in the scatterplot,
#### and visually identify them
#grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
#bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]
#
#
## You will need to complete this function imported from the ClassifyNB script.
## Be sure to change to that code tab to complete this quiz.
#clf = classify(features_train, labels_train)
#
#
#
#### draw the decision boundary with the text points overlaid
#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()#TODO
    clf.fit(features_train,labels_train)
    ### fit the classifier on the training features and labels
    #TODO

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)#TODO


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred,labels_test)#TODO
    return accuracy
    
if __name__=="__main__":
    import sys
    from class_vis import prettyPicture, output_image
    from prep_terrain_data import makeTerrainData
    from sklearn import tree
    import matplotlib.pyplot as plt
    import numpy as np
    import pylab as pl
    from sklearn.metrics import accuracy_score

    
    features_train, labels_train, features_test, labels_test = makeTerrainData()
    
    
    
    ### the classify() function in classifyDT is where the magic
    ### happens--fill in this function in the file 'classifyDT.py'!
    clf = tree.DecisionTreeClassifier(min_samples_split=50)
    clf = clf.fit(features_train,labels_train)
    
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred,labels_test)
    
    
    
    
    #### grader code, do not modify below this line
    
    prettyPicture(clf, features_test, labels_test)
    output_image("test.png", "png", open("test.png", "rb").read())