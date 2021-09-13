import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
#from sklearn.datasets import load_iris
# iris = load_iris()
# print(dir(iris))
#
# X_train, X_test, y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
#
# print(X_train)
#
# regressor = LogisticRegression()
# regressor.fit(X_train, y_train)
#
# print(iris.target[67]) #1
# #0 = setosa
# #1= versicolor
# #3= virginica
# print(iris.target[[10, 25, 50]]) #[setosa setosa versicolor]
#
# y_pred = regressor.predict([iris.data[67]])

# if y_pred[0] == 0:
#     print('The specie is Setosa')
# elif y_pred[0] == 0:
#     print('The specie is Versicolor')
# else:
#     print('The specie is Virginica')

df = pd.read_csv("C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\Iris.csv")
print(df.head(10))

df.drop('Id', axis=1, inplace=True)
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict([[6.4,2.9,4.3,1.3]])
print(y_pred)

SepalLengthCm = float(input('Enter Sepal Length in cm: '))
SepalWidthCm = float(input('Enter Sepal Width in cm: '))
PetalLengthCm = float(input('Enter Petal Length in cm: '))
PetalWidthCm = float(input('Enter Petal Width in cm: '))

pred_species = regressor.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm, PetalWidthCm]])
pred_species = np.reshape(pred_species,1)
print('The predicted specie is ', pred_species[0])