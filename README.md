# Binary-sentiment-classification
Implemented Perceptron and Logistic Regression and compared their performance on the sentiment classification task

### Structure:
- Execution.py
- Perceptron.py
- Logistic.py
- data.txt
- bog.txt
- test_data.csv

### Dataset Split:

- Initially the data is read as a Pandas dataframe as all_data and passed to split_dataset function.
- This function initially shuffles the entire dataset for unbiased distribution of good and bad sentiment examples
using
```python
df_permutated = all_data.sample(n=len(all_data), random_state=42)
```
- Setting the random_state assures the same shuffle pattern for every execution.
- Post this, the dataset is split in the ratio 80 : 20 for train_data and test_data
- The objective for splitting is to evaluate the performance of the machine learning model on new data: data not
used to train the model.
- The check for split correctness in both dataframes, the counts of positive and negative sentiment were printed
and found to be comparable assuring an unbiased training and evaluation.

For train data:
```
 1 692
 0 668
 Name: Label, dtype: int64
 ```
 
For test data:
```
 1 171
 0 169
 Name: Label, dtype: int64
```

The train data was further split into five folds. For each hyperparameter value, 4 folds were trained on those
values and one fold was used for testing. The average accuracy was printed and the weights and bias for the
best fold was set as _self.weight_ and _self.bias_ values which were used for predicting on unseen data.


### Feature Engineering:
For feature engineering, two methods were initially tried out: _bag of words_and _bag of words with n-grams_.

Pre-processing:
- All the numbers and special characters are removed from each word using
```
review = re.sub(’[^a-zA-Z]’, ’ ’,review)
```
- The characters are made lower_case for ease of comparison.
- The text is split into a list of words for construction of feature vector.
- All punctuations are removed.
- A custom list of stop words containing pronouns is made.
- All words in the above list are Lemmatized, that is, reduced to their root words and the stop words are
removed using
```
1 lemmatizer = WordNetLemmatizer()
2 review = [lemmatizer.lemmatize(w) for w in review if not w in stop_words]
```

A sentence
_the first " park " was a marvellous film , full of awe-inspiring sights , interesting characters , and genuine
thrills ._

Is reduced to:
_[’first’, ’park’, ’marvellous’, ’film’, ’full’, ’awe’, ’inspiring’, ’sight’, ’interesting’, ’character’, ’gen-
uine’, ’thrill’]_


### Creating Vocabulary

## Only bag of words:
Using a counter variable every word in the document is updated with its number of occurrence in the
document.

A list of words with minimum occurence over 10 times in the entire documents is retained. This list had
7605 words and this was used to construct a feature vector of length 7605 of 0s for vocab words not in
unseen example and 1s for vocab words present in unseen example respectively for every example in
data.

_the first " park " was a marvellous film , full of awe-inspiring sights , interesting characters , and genuine
thrills ._

Its feature vector is :
_[0, ....., 0, 1, 1, 0, 1, 0, ......0,0,1, 1, 0, 1], length = 7605_

**Size of vocabulary:** 7605

## bag of words with bigrams:
Here, list is created in the same way as for bag of words but every list element contains two consecutive
words each found in the document. The minimum occurrences was set as 5 and the feature vector was constructed similarly.

Example:

_["rich kid","chris klein","josh hartnett","gas station","leelee sobieski","soon take"]
_
**Size of vocabulary:** 6621

**Results using this process** were:
```
-------------Perceptron Performance-------------
        Training Accuracy Result!
        ***************
        Accuracy: 0.8529411764705882
        ***************
        
        Testing Accuracy Result!
        ***************
        Accuracy: 0.7205882352941176
        ***************
        
        Unseen Test Set Accuracy Result!
        ***************
        Accuracy: 0.6
        ***************
        
        -------------Logistic Function Performance-------------
        Training Accuracy Result!
        ***************
        Accuracy: 0.8911764705882353
        ***************
        
        Testing Accuracy Result!
        ***************
        Accuracy: 0.7235294117647059
        ***************
        
        Unseen Test Set Accuracy Result!
        ***************
        Accuracy: 0.8
        *************** 
 ```
 
 ## bag of words with 3-grams:
Here, list is created in the same way as for bag of words but every list element contains three consecutive
words each found in the document.
Here the minimum occurrences was set as 3 and the feature vector was constructed similarly.

**Size of vocabulary:** 2248

**Results using this process were:**
```
-------------Perceptron Performance-------------
        Training Accuracy Result!
        ***************
        Accuracy: 0.8058823529411765
        ***************
        
        Testing Accuracy Result!
        ***************
        Accuracy: 0.6411764705882353
        ***************
        
        Unseen Test Set Accuracy Result!
        ***************
        Accuracy: 0.6
        ***************
        
        -------------Logistic Function Performance-------------
        Training Accuracy Result!
        ***************
        Accuracy: 0.836764705882353
        ***************
        
        Testing Accuracy Result!
        ***************
        Accuracy: 0.6411764705882353
        ***************
        
        Unseen Test Set Accuracy Result!
        ***************
        Accuracy: 0.4
        ***************
```
Clearly, trigram performes poorly on unseen dataset due to its high dependence on order of words which
in turn depends on the size of training dataset. Thus 3-grams feature engineering tends to overfit

**Hence Bag of word was chosen as the feature engineering method due to its better performance in
Perceptron as well as Logistic Regression even though execution time was compromised which was
more than bi-gram and tri-gram by one minute.**

### Learning rate vs Accuracy for each Iteration value:

## Perceptron:
While training the perceptron model, the following hyper parameter values were used and cross-validate
upon:
_Learning Rate: [0.1, 0.3, 0.5, 0.7, 1]_ <br />
_Max_iterations: [1,5,8,10]_

![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m1%20p.png)
![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m5%20p.png)
![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m8%20p.png)
![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m10%20p.png)

## Logitstic Regression:
While training the logistic regression model, the following hyper parameter values were used and cross-validate upon:
_Learning Rate: [0.1, 0.5, 0.8, 1]_ <br />
_Max_iterations: [1,3,5,7,10]_

![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m1%20l.png)
![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m3%20l.png)
![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m5%20l.png)
![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m7%20l.png)
![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20m10%20l.png)

### Max_iteration vs avg_accuracy for constant learning rate

## Perceptron:
Learning rate set as 0.3:

![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20lr%20p.png)

##Logistic Regression:
Learning rate set as 0.8:

![plot](https://github.com/Sowmya-Iyer/Binary-sentiment-classification/blob/main/figures/SM21%20lr%20l.png)


MODEL results:

```
 -------------Perceptron Performance-------------
    Training Accuracy Result!
    ***************
    Accuracy: 0.9610294117647059
    ***************
    
    Testing Accuracy Result!
    ***************
    Accuracy: 0.8088235294117647
    ***************
```
```
-------------Logistic Function Performance-------------
    Training Accuracy Result!
    ***************
    Accuracy: 0.9713235294117647
    ***************
    
    Testing Accuracy Result!
    ***************
    Accuracy: 0.788235294117647
    ***************
```

By training-data performance, the evaluation result using both models seems comparable. <br \>
        However, for testing data, Perceptron performed much better, that is, Perceptron generalised to unseen data better than logistic regression. Hence, Perceptron, in my opinion, is more suitable for this task trained using the given data.      
