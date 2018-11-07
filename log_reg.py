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
import os

#map probabilities from 0-1 to 1-100 scale
def functionmap(A, B, a, b, val):
    """This method returns f(val) where f: [A,B] -> [a,b]"""
    return (val-A)*(b-a)/(B-A) + a

def findidx(elem, arr):
    """This method returns index of element in array (arrays operated are np.arrays here)"""
    for i in xrange(np.size(arr,0)):
        if False not in (arr[i]==elem):
            return i
    return -1

folders = ["dataset/facebook comments", "dataset/tweets"]

for folder in folders:
    print ("Accessing folder "+folder)
    #Get data from file
    for file_name in os.listdir(folder):
        if  "data.txt" in file_name and file_name[-3:]=="txt":

            average_accuracy = 0
            for _ in range(10):

                file_in = open(folder+"\\"+file_name,"r")
                #list of lines
                data = file_in.readlines()
                file_in.close()
                
                file_in = open(folder+"\\"+file_name,"r")
                #string
                data_string = file_in.read()
                print("Comment/Tweet count:", len(data))
                file_in.close()

                #Get labels of data
                file_in = open(folder+"\\"+file_name.rstrip("data.txt")+"label.txt","r")
                labels = file_in.read()
                file_in.close()


                #remove unnecessary spaces and other characters using RegEx on labels
                filtered_labels = list(filter(lambda x: not re.match(r'^\s*$', x), labels))

                #initialize vectorizer for our data
                vectorizer = CountVectorizer(
                    analyzer = 'word',
                    lowercase = False,
                )

                #fit and transform the data to a sparse matrix
                features = vectorizer.fit_transform(data)
                features_nd = features.toarray()

              

                #create random test and training sets based on given train_size; we randomly choose a seed
                X_train, X_test, y_train, y_test  = train_test_split(
                        features_nd, 
                        filtered_labels,
                        train_size=0.80, test_size=0.20,
                        random_state=random.randint(1,1000))

                #Set logistic regression model, using 'One vs Rest
                log_model = LogisticRegression(C=1e5, multi_class='ovr')
                log_model = log_model.fit(X=X_train, y=y_train)
                X=X_train

                #Probability estimates; each row corresponding to a tweet contains all probability estimates in order [P(N), P(O), P(P)]
                y_pred_prob = log_model.predict_proba(X_test)

                #Predicted labels
                y_pred = log_model.predict(X_test)

                #Initialize arrays for storing data by predicted labels
                Ns=[]
                Os=[]
                Ps=[]

                #Storing data (comment/tweet, P(X)) to array by predicted labels
                for i in xrange(len(y_pred_prob)):
                    if y_pred[i]=='N': #Predicted as N
                        Ns.append((data[findidx(X_test[i], features_nd)],y_pred_prob[i][0]))
                    elif y_pred[i]=='O': #Predicted as O
                        Os.append((data[findidx(X_test[i], features_nd)],y_pred_prob[i][1]))
                    elif y_pred[i]=='P': #Predicted as P
                        Ps.append((data[findidx(X_test[i], features_nd)],y_pred_prob[i][2]))

                #Sorting by probability estimates
                Ns.sort(key=lambda x:x[1])
                Ps.sort(key=lambda x:x[1])
                Os.sort(key=lambda x:x[1])

                #Continuous scale: N:1-33, O:33-66, P:66-99.99
                limits = [1,33,66,99.99]
                idx = 0
                for arr in [Ns, Os, Ps]:
                    minimum = arr[0][1]
                    maximum = arr[-1][1]
                    #map probability estimate to [1,99.99]
                    score = [(data, functionmap(minimum, maximum, limits[idx], limits[idx+1], val)) for data, val in arr]
                    idx+=1

                #Accuracy of One vs Rest Logistic Regression model on our data
                print("Accuracy:", accuracy_score(y_test, y_pred))
                average_accuracy+=accuracy_score(y_test, y_pred)

            print ("Example of tweets with scoring:")
            for comment,sc in score:
                print comment,sc

            print ("Average Accuracy after Cross validation:", average_accuracy/10.0)

            #remove punctuation
            data_string = data_string.translate(None,string.punctuation)

            #set of stop words
            stop_words = stopwords.words('english')

            #preprocess and tokenize data, remove stop words
            all_words = word_tokenize(p.clean(data_string))
            new_words=[]
            for w in all_words:
                if w not in stop_words:
                    new_words.append(w)

            #taking top 30 of frequency distribution of words
            words_freq_dist = nltk.FreqDist(new_words)
            top_occur = []
            for tup in words_freq_dist.most_common(30):
                    top_occur.append(tup[0].decode())            

            #check for nouns
            domain = []
            for i in top_occur:
                    blob = TextBlob(i)
                    noun = blob.noun_phrases
                    if(noun != []):
                        domain.append(noun[0])
            print("Top keywords:", domain)
            print "\n\n"
