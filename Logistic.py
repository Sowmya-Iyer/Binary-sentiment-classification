# from Perceptron import *
import re
import os
import pandas as pd
from collections import Counter
import numpy as np
from numpy import log,e,dot
from random import shuffle
from random import seed
bog=[]
with open("./bog.txt") as f:
    bog = f.read().split()

def split_train(data):
  train_data=[]
  test_data=[]
  df_permutated = data.sample(n=len(data), random_state=42)
  # df_permutated["Features"] = df_permutated["Text"].apply(lambda x : vectorizer(x,bog))

  for i in range(0,5): 
    test_size = int(len(df_permutated) / 5) 
    index= list(df_permutated.index)
    test_start= int((len(df_permutated)  / 5) *i)
    test_end =int(test_start+len(df_permutated)/5)
    test_index= index[test_start:test_end]
    train_index = [x for x in index if x not in test_index]
    # print(len(test_index))
    # print(len(train_index))
    df_test = df_permutated.loc[test_index]
    df_train = df_permutated.loc[train_index]
    train_data.append(df_train)
    test_data.append(df_test)
  return train_data,test_data

class Logistic():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the logistic function as instance
        attributes.
        """
        self.weights = []
        self.bias = []

    def sg(self,z):
        return 1/(1 + np.exp(-z))

    def feature_extraction(self):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in training.
        """
        return

    def cost_function(self, X, y, weights,bias):                 
        z = np.add(np.dot(X, weights),bias)
        predict_1 = np.dot(y,np.log(self.sg(z)))
        predict_0 = np.dot((1 - y), np.log(1 - self.sg(z)))
        loss=-(predict_1 + predict_0) 
        return loss


    def train(self, labeled_data, learning_rate=[0.8], max_epochs=[7]):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. This
        dataframe must have the label of the data points stored in a column called 'Label'. For example, the column
        labeled_data['Label'] must return the labels of every data point in the dataset. Additionally, this function
        should not return anything.

        The hyperparameters for training will be the learning rate and maximum number of epochs. Once you find the
        optimal values, update the default values for both the learning rate and max epochs keyword argument.

        The goal of this function is to train the logistic function on the labeled data. Feel free to code this
        however you want.
        """
        train,test=split_train(labeled_data)
        
        for epochs in max_epochs:         

          for lr in learning_rate:

            weights=[]
            bias=[]
            
            for fold in range(0,5):  
              X=train[fold]
              y=X.Label
              # X_train=X["Feature"]
              X_train=X.Features
              # print(len(X_train.loc[400]))
              # print(X_train)
              w=np.random.random_sample((len(bog),))
              b=0

              N=len(X_train)
              # print(X_train.shape)
              # print(N)
  
              for epoch in range(epochs):
                for i in X_train.index:
                  pred= self.sg(np.add(np.dot(X_train.loc[i],w),b))
                  gr_wrt_w = np.dot(X_train.loc[i],(pred - y[i]))
                  gr_wrt_b = pred - y[i]    
                  # pred= self.sg(dot(X_train.loc[i], w))
                  w = w - np.dot(lr,gr_wrt_w)
                  b = b - np.dot(lr,gr_wrt_b)
                  # loss+=(self.cost_function(X_train.loc[i], y[i], w,b))
              # num_epoch.append(epochs)
              
              weights.append(w)
              bias.append(b)

              # print(loss)
              # plt.plot(num_epoch, loss)
              # plt.title("Train set Logistic Loss - Fold"+str(fold))
              # plt.xlabel("Epochs")
              # plt.ylabel("Loss")
              # plt.show()
                # print("w",w)
                # print("b",b) 
            
            avg_acc=self.cross_val(test,weights,bias)
        return

    def cross_val(self,data,weights,bias):
        max=0
        best_acc=0
        total=0
        for fold in range(0,5):
            X_test=data[fold]['Features'].apply(np.array)
            y=data[fold].Label

            num=len(y)
            match=0

            for i in X_test.index:
              z=np.add(np.dot(X_test.loc[i],weights[fold]),bias[fold])
              pred= [1 if self.sg(z) > 0.5 else 0 ]
              if pred==y[i]:
                match+=1
            acc=float(match/num)  
            total+=acc  

            if(acc>max):
              max=acc
              best_fold=fold
            # print("Fold",fold+1,acc)
        avg_acc=total/5
        # print("Average_accuracy",avg_acc)
        if avg_acc > best_acc:
          self.weights=weights[best_fold]
          self.bias=bias[best_fold]
          best_acc= avg_acc
        return avg_acc

    def predict(self, data):
        """
        This function is designed to produce labels on some data input. The only input is the data in the form of a 
        pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the 
        predicted labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 1
        (positive) for each data point.

        The rest of the implementation can be fully customized.
        """
        # data["Features"]=data["Text"].apply(lambda x : vectorizer(x,bog))
        X_pred=data['Features'].apply(np.array)
        y=data.Label

        num=len(y)
        predicted_labels = []

        for i in X_pred.index:
          z=np.add(np.dot(X_pred.loc[i],self.weights),self.bias)
          pred= 1 if self.sg(z) > 0.5 else 0
          predicted_labels.append(pred)

        # print("predicted", predicted_labels)
        # predicted_labels=[1 for x in range(num)]
        return predicted_labels