#!/usr/bin/python
import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    index=[i[0] for i in sorted(enumerate(abs(numpy.subtract(predictions,net_worths))), key=lambda x:x[1])]
    cleaned_index=[]
    for a in range(int(len(index)*0.9)):
        cleaned_index.append(index[a])
    cleaned_data = []
    for b in cleaned_index:
        cleaned_data.append((ages[b],net_worths[b],predictions[b]))

    ### your code goes here
    
    
    return cleaned_data

