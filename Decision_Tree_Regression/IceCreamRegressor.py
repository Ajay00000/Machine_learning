import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('IceCreamData.csv')

x = data['Temperature'].values
y = data['Revenue'].values

# To Split data into Training set and testing set

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=22,test_size=0.2)

# Importing Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(xtrain.reshape(-1,1),ytrain.reshape(-1,1))

# Predicting The Results
ypred = regressor.predict(xtest.reshape(-1,1))

# Creating pandas data frame for predicting and actual values

df = pd.DataFrame({'Real Values':ytest.reshape(-1),'Predicting Values':ypred.reshape(-1)})

''' Visualizing The Result of Our Decision Tree Regression Model '''
plt.scatter(xtest,ytest,color="green")
plt.scatter(xtest,ypred,color="red")
plt.title("Temperature VS Revenue ( Decision Tree Regression )")
plt.xlabel(" Temperature ")
plt.ylabel(" Revenue ")
plt.show()

# =============================================================================
#  Visualizing the result of x only 
# =============================================================================

xgrid = np.arange(min(x),max(x),0.01)
xgrid = xgrid.reshape(len(xgrid),1)

plt.plot(xgrid,regressor.predict(xgrid),color='black')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

