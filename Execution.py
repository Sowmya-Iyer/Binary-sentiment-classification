import re
import os
import pandas as pd
from collections import Counter
import numpy as np
from random import shuffle
from random import seed
import matplotlib.pyplot as plt
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string
import codecs
import random
random.seed(0)

nltk.download('wordnet')

def vectorizer(text,bog):
  vector=[0 for i in range(0,len(bog))]
  for i in range(len(bog)):
      if any(n in text for n in list(bog[i].split())):
        vector[i]=1
      # print(bog[i])
  return vector

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',"you're","you've","you'll","you'd", 'your','yours','yourself','yourselves', 'he','him', 'his','himself','she',"she's",'her','hers','herself', 'it',"it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when','where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own','same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",'now', 'd', 'll', 'm','o','re', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def save_list(lines, filename):
# convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = codecs.open(filename, 'w', encoding='utf8')
    # write text
    file.write(data)
    # close file
    file.close()

def review_to_words(raw_review): 
    review = raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    review= [re_punc.sub('', w) for w in review]
    review = [word for word in review if len(word) > 1]
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in stop_words]
    # print(review)
    # review = nltk.word_tokenize(review)
    return review

def generate_ngrams(s, n=1):
    tokens = s
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def add_doc_to_vocab(dataframe, vocab):
    # doc = list(dataframe["Cleaned_Text"])
    # tokens=review_to_words(doc)
    tokens=list(dataframe["Text"])
    # print(tokens)
    for i in tokens:
      # c=Counter(generate_ngrams(i, 2))
      c=Counter(i)
      vocab.update(c)
      # print(c)
    min_occurance = 10
    bog = [k for k,c in vocab.items() if c > min_occurance]
    # save_list(bog, 'vocab.txt')
    return bog

def process_docs(dataframe, vocab):
# walk through all files in the folder
    bog=add_doc_to_vocab(dataframe, vocab)
    return bog
# vocab = Counter()
# bog=process_docs(df, vocab)

def split_dataset(all_data):
    train_data = []
    test_data = []
    """
    This function will take in as input the whole dataset and you will have to program how to split the dataset into
    training and test datasets. These are the following requirements:
        -The function must take only one parameter which is all_data as a pandas dataframe of the raw dataset.
        -It must return 2 outputs in the specified order: train and test datasets
        
    It is up to you how you want to do the splitting of the data.
    """
    df_permutated = all_data.sample(n=len(all_data), random_state=42)
    # print(df_permutated.index)
    
    train_data=df_permutated.sample(frac=0.8,random_state=42) #random state is a seed value
    test_data=df_permutated.drop(train_data.index)
    # print(train_data.Label.value_counts())
    # print(test_data.Label.value_counts())
  
    return train_data, test_data

"""
This function should not be changed at all.
"""
def eval(o_train, p_train, o_val, p_val, o_test, p_test):
    print('\nTraining Accuracy Result!')
    accuracy(o_train, p_train)
    print('\nTesting Accuracy Result!')
    accuracy(o_val, p_val)
    print('\nUnseen Test Set Accuracy Result!')
    accuracy(o_test, p_test)

"""
This function should not be changed at all.
"""
def accuracy(orig, pred):
    num = len(orig)
    if (num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
        o_label = orig[i]
        p_label = pred[i]
        if (o_label == p_label):
            match += 1
    print('***************\nAccuracy: '+str(float(match) / num)+'\n***************')


if __name__ == '__main__':
    """
    The code below these comments must not be altered in any way. This code is used to evaluate the predicted labels of
    your models against the ground-truth observations.
    """
    from Perceptron import Perceptron
    from Logistic import Logistic
    all_data = pd.read_csv('./data.csv', index_col=0)
    all_data['Text'] = all_data['Text'].apply(lambda x : review_to_words(x))
    
    vocab=Counter()
    bog=add_doc_to_vocab(all_data,vocab)
    save_list(bog, 'bog.txt')
    all_data["Features"] = all_data["Text"].apply(lambda x : vectorizer(x,bog))
    # print(len(bog))


    train_data, test_data = split_dataset(all_data)

    test_data_unseen = pd.read_csv('./test_data.csv', index_col=0)
    test_data_unseen['Text']=test_data_unseen['Text'].apply(lambda x : review_to_words(x))
    test_data_unseen["Features"] = test_data_unseen["Text"].apply(lambda x : vectorizer(x,bog))

    perceptron = Perceptron()
    logistic = Logistic()

    perceptron.train(train_data)
    logistic.train(train_data)

    # print(type(train_data))
    # print(type(test_data))
    # print(type(test_data_unseen))

    predicted_train_labels_perceptron = perceptron.predict(train_data)
    predicted_test_labels_perceptron = perceptron.predict(test_data)
    predicted_test_labels_unseen_perceptron = perceptron.predict(test_data_unseen)

    predicted_train_labels_logistic = logistic.predict(train_data)
    predicted_test_labels_logistic = logistic.predict(test_data)
    predicted_test_labels_unseen_logistic = logistic.predict(test_data_unseen)

    print('\n\n-------------Perceptron Performance-------------\n')
    # This command also runs the evaluation on the unseen test set
    eval(train_data['Label'].tolist(), predicted_train_labels_perceptron, test_data['Label'].tolist(),
         predicted_test_labels_perceptron, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_perceptron)
    print('\n\n-------------Logistic Function Performance-------------\n')
    # This command also runs the evaluation on the unseen test
    eval(train_data['Label'].tolist(), predicted_train_labels_logistic, test_data['Label'].tolist(),
         predicted_test_labels_logistic, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_logistic)