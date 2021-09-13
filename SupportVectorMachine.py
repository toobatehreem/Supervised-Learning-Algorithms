from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd

iris = load_iris()
print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

df['target'] = iris.target
print(df.head())

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

df0 = df[df.target ==0]
df1 = df[df.target ==1]
df2 = df[df.target ==2]

print(df0.head())
print(df1.head())
print(df2.head())

plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='.')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='.')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

X = df.drop(['target', 'flower_name'], axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = SVC() #C = 1.0 , maybe increasing C(regularization) will decrease the score
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(y_pred)