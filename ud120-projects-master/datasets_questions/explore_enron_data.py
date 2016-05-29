#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import os
import pickle
import re
os.chdir(os.path.dirname(os.path.realpath(__file__)))
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print enron_data.items()[0][1]
interested_number=0
for people in enron_data.items():
    if people[1]["poi"]==True:
        interested_number+=1

print "Feature number",len(enron_data.items()[0][1])

names=open(r"D:\github\machine_learning\ud120-projects-master\final_project/poi_names.txt","r")
print len(names.readlines())
names.seek(0)

def find_people(dataset=enron_data,name="PRENTICE JAMES"):
    for people in dataset.items():
        if re.search(name, people[0], re.IGNORECASE):
            print "find you,",people[0],"!"
            print people#,"total stock belonging:",people[1]["total_stock_value"]
            #for feature in people[1]:
            #    if "stock" in feature:
            #        print feature
def find_feature(dataset=enron_data,name="PRENTICE JAMES",feature="email"):
    related_features=set()
    related_people=set()
    for people in dataset.items():
        if re.search(name, people[0], re.IGNORECASE):
            related_people.add(people[0])
            for one in people[1]:
                if re.search(feature, one, re.IGNORECASE):
                    related_features.add(one)
    print "The realated features are",related_features,"of ",related_people
                #for feature in people[1]:
                #    if "stock" in feature:
                #        print feature
def get_feature(dataset=enron_data,name="PRENTICE JAMES",feature="email"):
    for people in dataset.items():
        if people[0]==name:
            return people[1][feature]
if __name__== "__main__":
    number_quantified_salary=0
    for people in enron_data.items():
        try:
            int(people[1]["salary"])
            number_quantified_salary +=1
        except ValueError:
            pass
    number_email=0
    for people in enron_data.items():
        if re.search(".com", people[1]["email_address"], re.IGNORECASE):
            number_email +=1
