import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv('Position_Salaries.csv')

x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

# Import Random Forest Regression from sklearn to fited 

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

# Predict the values 
regressor.predict([[6.5]])

# Visualising the results 

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title("Level VS Salary (Random Forest Regression )")
plt.xlabel("LEVEL")
plt.ylabel("SALARY")
plt.show()

# Visualizing With Xgrid

xgrid = np.arange(min(x),max(x),0.01)
xgrid = xgrid.reshape((len(xgrid),1))
plt.scatter(x,y,color='red')
plt.plot(xgrid,regressor.predict(xgrid),color='blue')
plt.title("Level VS Salary (Random Forest Regression )")
plt.xlabel("LEVEL")
plt.ylabel("SALARY")
plt.show()