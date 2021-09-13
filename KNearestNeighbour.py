from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

features = iris.data
labels = iris.target

print(features[0], labels[0])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_pred =  model.predict(X_test)
print(y_pred)