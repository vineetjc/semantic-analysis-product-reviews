from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re, random, string
from sklearn.metrics import accuracy_score
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer
import preprocessor as p
from nltk.corpus import stopwords
from textblob import TextBlob 

#Read tweets data
fin = open("../dataset/facebook/fb_data.txt","r")
datastr = fin.read()
data = fin.readlines()
#print len(data)
fin.close()

#Read tweets label
fin = open("../dataset/facebook/fb_label.txt","r")
datal = fin.read()
fin.close()
filtered = list(filter(lambda x: not re.match(r'^\s*$', x), datal))

# vectorizer = CountVectorizer(
#     analyzer = 'word',
#     lowercase = False,
# )
# features = vectorizer.fit_transform(
#     data
# )
# features_nd = features.toarray()

# X_train, X_test, y_train, y_test  = train_test_split(
#         features_nd, 
#         filtered,
#         train_size=0.80,
#         random_state=random.randint(1,1000))

# log_model = LogisticRegression(multi_class='ovr')
# log_model = log_model.fit(X=X_train, y=y_train)
# y_pred = log_model.predict(X_test)

# print(accuracy_score(y_test, y_pred))



datastr = datastr.translate(None,string.punctuation)
stop_words = stopwords.words('english')
all_words = word_tokenize(p.clean(datastr))
new_words=[]
for w in all_words:
    if w not in stop_words:
        new_words.append(w)
#print(stop_words)

new_words = nltk.FreqDist(new_words)
top_occur = []
for tup in new_words.most_common(30):
	top_occur.append(tup[0].decode())

# stri = ""
# for i in domain:
# 	stri = stri+" " + i
# text = word_tokenize(stri)
# print(nltk.pos_tag(text))
#print(top_occur)
domain = []
for i in top_occur:
	blob = TextBlob(i)
	noun = blob.noun_phrases
	if(noun != []):
		print(noun[0])
		domain.append(noun[0])
#print(domain)


