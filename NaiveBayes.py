import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\titanic.csv')
print(df.head(10))

df.drop(['Embarked','PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)
# df.dropna(inplace=True)
print(df.describe())
print(df.head())

inputs = df.drop('Survived', axis='columns')
target = df['Survived']

print(inputs)

dummies = pd.get_dummies(inputs.Gender)
print(dummies.head())

inputs = pd.concat([inputs,dummies], axis=1)
print(inputs.head())

inputs.drop(['Gender'], axis=1, inplace=True)
print(inputs.head())

print(inputs.columns[inputs.isna().any()])

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.head())

print(inputs.columns[inputs.isna().any()])

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(y_pred)



cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

pclass = int(input('Enter Pclass: '))
age = int(input('Enter age: '))
fare = float(input('Enter fare: '))
female = int(input('Enter 1 for female and 0 for male: '))
male = int(input('Enter 1 for male and 0 for female: '))

pred = model.predict([[pclass,age,fare,female,male]]) #Pclass, Age, Fare, Gender

if pred[0] == 0:
    print("The person didn't survived")
else:
    print("The person survived")