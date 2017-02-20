import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# Read data
dataframe = pd.read_fwf('brain_body.txt')
print(dataframe)
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# train model on data
body_regression = linear_model.LinearRegression()
body_regression.fit(x_values, y_values)

# visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_regression.predict(x_values))
plt.show()
