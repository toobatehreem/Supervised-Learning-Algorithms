import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math

titanic = pd.read_csv("C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\titanic.csv")
print(titanic)

print(titanic['PassengerId'].count)

# sns.countplot(x='Survived', data = titanic)
# plt.show()
#
# sns.countplot(x='Survived', hue='Gender', data=titanic)
# plt.show()
#
# sns.countplot(x='Survived', hue='Pclass', data=titanic) #pclass = passenger class
# plt.show()
#
# sns.countplot(x='Survived', hue='Age', data=titanic)
# plt.show()
#
# titanic['Age'].plot.hist()
# plt.show()
#
# titanic['Fare'].plot.hist(bins=20, figsize=(10,10))
# plt.show()
#
# print(titanic.info())
#
# sns.countplot(x='SibSp', data = titanic) #siblings and spouse
# plt.show()

print(titanic.isnull())

print(titanic.isnull().sum())

sns.heatmap(titanic.isnull(), yticklabels=False, cmap='Blues')
#plt.show()

sns.boxplot(x='Pclass', y='Age', data=titanic)
#plt.show() #passengers in class 1 and 3 are older than passengers travelling in class 3

titanic.drop('Cabin', axis=1, inplace=True)
print(titanic.head(10))
sns.heatmap(titanic.isnull(), yticklabels=False, cmap='Blues')
#plt.show()

titanic.dropna(inplace=True)
print(titanic.head(10))
sns.heatmap(titanic.isnull(), yticklabels=False, cmap='Blues')
#plt.show()

print(titanic.isnull().sum())

gender = pd.get_dummies(titanic['Gender'],drop_first=True) #because it is categorical and logistic regression can only work on categorical data
print(gender)

embark = pd.get_dummies(titanic['Embarked'],drop_first=True)
print(embark)

Pcl = pd.get_dummies(titanic['Pclass'],drop_first=True)
print(Pcl)

titanic = pd.concat([titanic, gender, embark, Pcl], axis=1)
print(titanic)

titanic.drop(['Gender', 'Embarked', 'PassengerId','Pclass', 'Name', 'Ticket'], axis=1, inplace=True)
print(titanic)

X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))
