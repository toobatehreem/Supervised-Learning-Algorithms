import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# df = pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\salaries.csv')
#
# print(df.head(10))
#
# inputs = df.drop('salary_more_then_100k', axis='columns')
# target = df['salary_more_then_100k']
#
# print(inputs)
# print(target)
#
# #machine learning algortithms can only work on numbers and they can't understand labels, hence use encoders ro convert it into numbers
#
# le_company = LabelEncoder()
# le_job = LabelEncoder()
# le_degree = LabelEncoder()
#
# inputs['company_n'] = le_company.fit_transform(inputs['company'])
# inputs['job_n'] = le_job.fit_transform(inputs['job'])
# inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
#
# print(inputs.head(10))
# inputs_n = inputs.drop(['company', 'job', 'degree'], axis=1)
#
# print(inputs_n)
#
# model = tree.DecisionTreeClassifier()
# model.fit(inputs_n, target)
#
# print(model.score(inputs_n,target))
#
# pred = model.predict([[1,2,0]])
#
# print(pred)

df = pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\titanic.csv')
print(df.head(10))

df.drop(['Embarked','PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)
df.dropna(inplace=True)
print(df.describe())
print(df.head())

inputs = df.drop('Survived', axis='columns')
target = df['Survived']

print(inputs.describe())
print(inputs)
print(target)

le_gender = LabelEncoder()

inputs['gender_n'] = le_gender.fit_transform(inputs['Gender'])

inputs_n = inputs.drop(['Gender'], axis=1)
print(inputs_n.head())

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

print(model.score(inputs_n,target))

pclass = int(input('Enter Pclass: '))
age = int(input('Enter age: '))
fare = float(input('Enter fare: '))
gender = int(input('Enter 1 for male and 0 for female: '))

pred = model.predict([[pclass,age,fare,gender]]) #Pclass, Age, Fare, Gender

if pred[0] == 0:
    print("The person didn't survived")
else:
    print("The person survived")