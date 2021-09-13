import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

fruits= pd.read_table('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\fruit_data.txt')
print(fruits.head())

print(fruits.shape)
print(fruits['fruit_name'].unique())
print(fruits.groupby('fruit_name').size())

sns.countplot(fruits['fruit_name'], label='Count')
#plt.show()

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#normalizing the data so that each variable will have a similar range, e.g mass is in double/triple values, height is in single value whereas color_score is not even in single values
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(y_pred)
print('Accuracy of Logistic Regression Classfier on training data is: {:.2f}'.format(regressor.score(X_train, y_train)))
print('Accuracy of Logistic Regression Classfier on testing data is: {:.2f}'.format(regressor.score(X_test, y_test)))
print('\n')

from sklearn.tree import DecisionTreeClassifier

dtclf = DecisionTreeClassifier()
dtclf.fit(X_train, y_train)

y_pred = dtclf.predict(X_test)
print(y_pred)
print('Accuracy of Decision Tree Classfier on training data is: {:.2f}'.format(dtclf.score(X_train, y_train)))
print('Accuracy of Decision Tree Classfier on testing data is: {:.2f}'.format(dtclf.score(X_test, y_test)))
print('\n')

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)
print('Accuracy of KNN Classfier on training data is: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN Classfier on testing data is: {:.2f}'.format(knn.score(X_test, y_test)))
print('\n')

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(y_pred)
print('Accuracy of gnb Classfier on training data is: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of gnb Classfier on testing data is: {:.2f}'.format(gnb.score(X_test, y_test)))
print('\n')

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print(y_pred)
print('Accuracy of SVM Classfier on training data is: {:.2f}'.format(svc.score(X_train, y_train)))
print('Accuracy of SVM Classfier on testing data is: {:.2f}'.format(svc.score(X_test, y_test)))

#Confusion matrix for knn classifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pred = knn.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


