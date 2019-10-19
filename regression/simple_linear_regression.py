# Simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

# Fitting simple linear regression model to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predict test results
predictions = regressor.predict(x_test)

# visualize training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()

# visualize test set results
plt.scatter(x_test, y_test, color='yellow')
plt.plot(x_train, regressor.predict(x_train), color='green')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()
