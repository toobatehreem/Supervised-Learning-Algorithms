import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\spam.csv')
print(df.head(10))

print(df.groupby('Category').describe())

df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2, random_state=0)

#using countVectorizer to convert message words as numbers

vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train.values)

print(vectorizer.get_feature_names()) #all the unique words in email messages
print(X_train_count.toarray()) #which message contains which unique words

model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = ['Hey mohan, can we get together to watch football game tomorrow?','Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!']

emails_count = vectorizer.transform(emails)
ans_pred = model.predict(emails_count)
print(ans_pred)

X_test_count = vectorizer.transform(X_test)
print(model.score(X_test_count, y_test))

#using sklearn pipeline api to reduce the inconsistency created by transforming the data everytime

clf = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())]) #classifer to convert text into count vectorizer then apply multinomial naive bayes

#now we can directly feed the text into our model

training = clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

print(clf.predict(emails))