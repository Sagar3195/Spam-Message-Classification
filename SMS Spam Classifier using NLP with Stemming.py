### SMS Spam Classifier using NLP with Stemming

import pandas as pd
import numpy as np

#Loading spam classifier datasets
df = pd.read_csv("spam.csv", encoding='latin-1')

print(df.head())

print(df.columns)

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)

print(df.head())

print(df.iloc[2])

print(df['class'].value_counts())

print(df.shape)

#Checking missing values in dataset
print(df.isnull().sum())
#### We can see that there is no missing values in dataset

print(df.dtypes)

##Data cleanning and preprocessing

import nltk
import re

#Stopwords is already downloaded
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem   import PorterStemmer
from nltk.stem   import WordNetLemmatizer

#create variable for stemming and lemmatization
stemmer = PorterStemmer()
#lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review) #joining all the words into sentence'
    corpus.append(review)

#print(corpus)

print(len(corpus))

#Now creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#We set max feature 5000 bcoz some of the words will not more frequently present,
#It may be one or two time frequently present
vectorizer = CountVectorizer(max_features= 5000)

#here we get independent feature
X = vectorizer.fit_transform(corpus).toarray()
print(X.shape)

print(X.ndim)

#now dependent variable
y = pd.get_dummies(df['class'])

print(y.head())

y = y.iloc[:,1].values


#here spam is 0 and ham is 1 ie not spam 
y ### Here we get dependent variable

#now splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


print(x_train.shape, x_test.shape)

#now training the model using Naive Bayer classifier
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

#fit the model and predict the model
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

#now checking performance of the model and accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy of the model: ",accuracy)

cm = confusion_matrix(y_test, y_predict)
print("Confusion matrix of model: \n",cm)

import joblib
joblib.dump(model, 'spam_ham_message.pkl')
joblib.dump(vectorizer, 'transform.pkl')