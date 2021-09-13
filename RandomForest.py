import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#term ensemble is used when you're using multiple algorithms to predict the outcomes(building multiple decision trees to get an outcome)

digits = load_digits()
print(dir(digits))

print(digits.data[67])

#for i in range(4):
    #plt.matshow(digits.images[i])
    # plt.show()

df = pd.DataFrame(digits.data)

print(df.head())

df['target'] = digits.target #adding a new column in our data
#target is nothing but the actual mapping of the data, like image 0 = digit 0, image 8 = digit 8 and so on
print(df.head(13))

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis=1), digits.target, test_size=0.2, random_state=0)

print(len(X_train))
print(len(X_test))

#model = RandomForestClassifier(n_estimators=50) #more number of trees will increase the score
#max score = 97%, using n_estimators = 50, when increasing estimators after 50,the score is decresing
model = RandomForestClassifier()
model.fit(X_train, y_train) #n_estimators = 10, means it has used 10 random trees

print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()