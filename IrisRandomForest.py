import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\Iris.csv')

print(df.head())

df.drop(['Id'], axis=1, inplace=True)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Species'], axis=1), df['Species'], test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_pred = model.predict(X_test)

plt.figure(figsize=(10,7))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.xlabel = 'Predicted'
plt.ylabel = 'Truth'
plt.title = 'Predicted vs Truth'
plt.show()

SepalLengthCm = float(input('Enter Sepal Length in cm: '))
SepalWidthCm = float(input('Enter Sepal Width in cm: '))
PetalLengthCm = float(input('Enter Petal Length in cm: '))
PetalWidthCm = float(input('Enter Petal Width in cm: '))

pred_species = model.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm, PetalWidthCm]])
pred_species = np.reshape(pred_species,1)
print('The predicted specie is ', pred_species[0])
