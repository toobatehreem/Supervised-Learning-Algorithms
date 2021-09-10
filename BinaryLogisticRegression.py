import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df= pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\insurance.csv')

print(df.head(10))
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.show()

X = df['age'].values.reshape(-1,1)
y = df['bought_insurance'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

print(regressor.score(X_test, y_test))
print(X_test)
print(regressor.predict_proba(X_test)) #(not buy, buy) probabilities
y_pred = regressor.predict(X_test)

ages = int(input('Enter an age: '))
ages = [[ages]]
is_buying = regressor.predict(ages)
is_buying = np.reshape(is_buying,1)

if is_buying[0]== 1:
    print('The person will buy the insurance: YES')
else:
    print('The person will buy the insurance: NO')