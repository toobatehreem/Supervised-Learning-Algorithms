import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\Weather.csv")

print(df.shape)

print(df.describe())

df.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

# plt.figure(figsize=(15,10))
# plt.tight_layout() #automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
sns.distplot(df['MaxTemp'])
plt.show()

#Data Splicing

X = df['MinTemp'].values.reshape(-1,1) #shaping X with n(not known) rows and 1 column
y = df['MaxTemp'].values.reshape(-1,1)

print(X)
print(np.shape(X))
print(np.shape(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Setting random_state a fixed value will guarantee that same sequence of random numbers are generated each time you run the code.

print(X_train)
print(np.shape(X_train)) #having random 80% of the rows of the data

print(X_test)
print(y_train)
print(np.shape(y_train)) #having random 80% of the rows of the data

print(y_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

#Finding the intercept and coefficient where the line best fits
print('Intercept ', regressor.intercept_)
print('Coefficient ', regressor.coef_) #slope

#coefficient tells that for every 1 unit change in minimun temperature there is a 0.92 change in max temp

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