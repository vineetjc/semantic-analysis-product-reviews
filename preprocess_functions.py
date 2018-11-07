import os, random, gzip
import gensim
from sklearn.feature_extraction.text import *
from sklearn.externals import joblib
from sklearn.datasets import dump_svmlight_file
import datetime
import numpy as np
import shutil
from math import ceil

def convert_txt_to_gzip(fname_in):
    """This method gives a gzip file for a text file (used for read_input())"""
    with open(fname_in, 'rb') as f_in:
        with gzip.open(fname_in+".gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return fname_in+".gz" #passes as argument into read_input() function call


def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    print input_file
    documents = []
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 100 == 0):
                print "read "+str(i)+" reviews"
            # do some pre-processing and return list of words for each review
            # text
            documents.append(u' '.join(gensim.utils.simple_preprocess(line)).encode(encoding='utf-8', errors='ignore').strip())
    return documents

def tweet_to_vectors(ratio=20):
    "This method converts tweets/comments to matrices using Tfidf Vectorizer"
    folders = ["dataset\\facebook comments", "dataset\\tweets"]
    for folder_name in folders:
        for file_name in os.listdir(folder_name):
            if 'data.txt'in file_name and file_name[-3:]=='txt':
                with open(folder_name+"\\"+file_name, "r") as data_file:
                    #load data from file
                    tweets = data_file.readlines()
                    tweet_count = len(tweets)
                    test_count = int(ceil(ratio*tweet_count/100.0))

                    #make new directory for every time we test
                    newdir = "Test-"+datetime.datetime.now().strftime("%d-%m-%Y-%Hh%Mm%Ss")
                    if not os.path.isdir(folder_name+"\\"+newdir):
                        os.makedirs(folder_name+"\\"+newdir)

                    #files for writing tweets into separate files (to verify)
                    train_tweets_file = open(folder_name+"\\"+newdir+"\\trainingset.txt", "w")
                    test_tweets_file = open(folder_name+"\\"+newdir+"\\testingset.txt", "w")

                    #loading label matrix file
                    label_matrix_file = open(folder_name+"\\Y.txt", "r")
                    train_labels = label_matrix_file.readlines()

                    #files for writing labels corresponding to the training and testing sets
                    train_label_matrix_file = open(folder_name+"\\"+newdir+"\\Y_train_matrix.txt", "w")
                    test_label_matrix_file = open(folder_name+"\\"+newdir+"\\Y_test_matrix.txt", "w")

                    #convert text file to gzip, return list of preprocessed tweets
                    train_tweets = read_input(convert_txt_to_gzip(folder_name+"\\"+file_name))

                    #for verification
                    if len(train_tweets)!=tweet_count:
                        raise Exception("Missed out preprocessing some tweets")
                    if tweet_count!=len(train_labels):
                        raise Exception("Error in label matrix dimension or tweet count")

                    #randomly picking tweets (and corresponding labels) to move into new arrays
                    test_tweets = []
                    test_labels = []
                    temp = tweet_count
                    for i in xrange(test_count):
                        idx = random.randint(0, temp-1)
                        test_tweets.append(train_tweets[idx])
                        train_tweets.pop(idx)
                        test_labels.append(train_labels[idx])
                        train_labels.pop(idx)
                        temp-=1 #train_tweets size reduces by 1 on every iteration of this for loop

                    #write contents into respective files
                    for line in train_tweets:
                        train_tweets_file.write(line+"\n")
                    for line in test_tweets:
                        test_tweets_file.write(line+"\n")
                    for line in train_labels:
                        train_label_matrix_file.write(line)
                    for line in test_labels:
                        test_label_matrix_file.write(line)
                    train_tweets_file.close()
                    test_tweets_file.close()
                    train_label_matrix_file.close()
                    test_label_matrix_file.close()

                    #Vectorizer on testing set
                    X = TfidfVectorizer(norm = 'l2', max_features=150, encoding='latin-1')
                    Z = X.fit_transform(test_tweets)
                    feature_names = X.get_feature_names()
                    doc = 0
                    feature_index = Z[doc,:].nonzero()[1]
                    tfidf_scores = zip(feature_index, [Z[doc, x] for x in feature_index])
                    idf = X.idf_
                    Y_testmat = np.loadtxt(folder_name+"\\"+newdir+"\\Y_test_matrix.txt")
                    dump_svmlight_file(Z, Y_testmat, folder_name+"\\"+newdir+"\\"+"Test.txt", zero_based=True, comment=None, query_id=None, multilabel=True)

                    #Vectorizer on training set
                    X = TfidfVectorizer(norm = 'l2', max_features=150, encoding='latin-1')
                    Z = X.fit_transform(train_tweets)
                    feature_names = X.get_feature_names()
                    doc = 0
                    feature_index = Z[doc,:].nonzero()[1]
                    tfidf_scores = zip(feature_index, [Z[doc, x] for x in feature_index])
                    idf = X.idf_
                    Y_trainmat = np.loadtxt(folder_name+"\\"+newdir+"\\Y_train_matrix.txt")
                    dump_svmlight_file(Z, Y_trainmat, folder_name+"\\"+newdir+"\\"+"Train.txt", zero_based=True, comment=None, query_id=None, multilabel=True)

if __name__=="__main__":
    tweet_to_vectors()
