import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

digits = load_digits()

print(dir(digits))

print(digits.data[0]) #first data in the form of numpy array

plt.gray()
plt.matshow(digits.images[0]) #first data in the form of an image
plt.show()

# for i in range(5):
#     plt.matshow(digits.images[i]) #first 5 images in the data
#     plt.show()

print(digits.target[0:5])

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

print(regressor.score(X_test, y_test))

# y_pred = regressor.predict(X_test)
#
# print(y_pred)

plt.matshow(digits.images[67])
plt.show()

print(digits.target[67])

y_pred = regressor.predict([digits.data[67]])

print(y_pred[0])

y_pred = regressor.predict(digits.data[0:5])

print(y_pred)

'''Confusion Matrix: To know where my model is failing'''

y_pred = regressor.predict(X_test)

cm = confusion_matrix(y_test, y_pred) #y_test = truth, y_pred = the values we've got

print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()