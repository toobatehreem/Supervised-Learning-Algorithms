import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\data.csv')

print(df.head(10))

df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

X = df['Hours'].values.reshape(-1,1)
y = df['Scores'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

df_predicted = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(df_predicted)

df1 = df_predicted.head(25)
df1.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth='2')
plt.show()


print('Mean Absolute Error ',metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

hours = int(input('Enter the number of hours you studied: '))
hour = [[hours]]
score_pred = regressor.predict(hour)
print('Predicted Score= {} '.format(score_pred[0]))