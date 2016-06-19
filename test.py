from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
string1="hi katie the self driving car will be late best sebastian"
string2="hi sebastian the machine learning will be great great great best katie"
string3="hi katie the machine learning will be most excellent"
email_list=[string1,string2,string3]
bag_of_words=vectorizer.fit(email_list)
bag_of_words=vectorizer.transform(email_list)
for a in bag_of_words:
    print a
feature_words = vectorizer.get_feature_names() 
dense = bag_of_words.todense()
print ''
print 'what are the feature words?'
print feature_words

print ''
print 'how many times do the feature words occur in each string?'
print dense


print ''
print 'what number is "great"?'
print vectorizer.vocabulary_.get('great')