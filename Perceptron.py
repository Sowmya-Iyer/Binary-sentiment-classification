import re
import os
import pandas as pd
from collections import Counter
import numpy as np
from random import shuffle
from random import seed


bog=[]
with open("./bog.txt") as f:
    bog = f.read().split()
    
def split_train(data):
  train_data=[]
  test_data=[]
  df_permutated=data.sample(n=len(data), random_state=42)
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


class Perceptron():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the perceptron as instance attributes.
        """
        self.weights = []
        self.bias = []

    def train(self, labeled_data, learning_rate=[0.3], max_iter=[5]):
        
        train,test=split_train(labeled_data)
       
        for max_iteration in max_iter:
          
          # print("iter",max_iteration)
          for lr in learning_rate:
            # print("lr",lr)
            weights=[]
            bias=[]
            for fold in range(0,5):  
              y=train[fold].Label
              # X_train=X["Feature"]
              X_train=train[fold].Features

              # print(len(X_train.loc[400]))
              # print(X_train)
              w=np.zeros(len(bog))
              b=0
  
              for iter in range(max_iteration):
                for i in list(X_train.index):
                  result=np.add(np.dot(X_train.loc[i],w),b)
                  pred= 1 if result>0 else -1
                  actual= 1 if y.loc[i]>0 else -1

                  if pred*actual<=0:
                    w = w + np.dot(lr,np.dot(actual,X_train.loc[i]))
                    b=b+lr*actual
                weights.append(w)
                bias.append(b)
                # print("w",w)
                # print("b",b) 
            avg_acc,is_best=self.cross_val(test,weights,bias)
            # ila=[]
            # ila.append(max_iteration)
            # ila.append(lr) 
            # ila.append(avg_acc)
            # ILA.append(ila)
            # if is_best:
            #   self.lr=lr
            #   self.max_iter=max_iteration
          # print(ILA)
          # print

        return
    def cross_val(self,data,weights,bias):
        max=0
        best_acc=0
        total=0
        for fold in range(0,5):
            X_test=data[fold]['Features']
            y=data[fold].Label

            num=len(y)
            match=0

            for i in X_test.index:
              result=np.add(np.dot(X_test.loc[i],weights[fold]),bias[fold])
              pred= 1 if result>0 else -1
              actual= 1 if y.loc[i]>0 else -1

              if pred==actual:
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
          return avg_acc, 1
        return avg_acc,0 

    def predict(self, data):
        
        """
        This function is designed to produce labels on some data input. The first input is the data in the form of a 
        pandas dataframe. 
        
        Finally, you must return the variable predicted_labels which should contain a list of all the 
        predicted labels on the input dataset. This list should only contain integers that are either 0 (negative) or 1
        (positive) for each data point.
        
        The rest of the implementation can be fully customized.
        """
        # data["Features"]=data["Text"].apply(lambda x : vectorizer(x,bog))
        X_pred=data['Features']
        y=data.Label

        num=len(y)
        predicted_labels = []
      

        for i in X_pred.index:
          result=np.add(np.dot(X_pred.loc[i],self.weights),self.bias)
          pred= 1 if result>=0 else 0
          predicted_labels.append(pred)
 
        # if "list" in str(type(data)):
        #   for fold in range(X=data.Text.apply(lambda x : vectorizer(x,bog))

        return predicted_labels
