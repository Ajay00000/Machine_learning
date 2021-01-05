import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values
# Import the Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(x,y)
# Predicting The Result
regressor.predict([[6.5]])

''' Visualizing The results '''

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Level VS Salary (Decision Tree Regression)')
plt.xlabel("Level")
plt.ylabel("Sallary")
plt.show()