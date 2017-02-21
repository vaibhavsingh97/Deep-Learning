import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from pylab import figure

# Read dataset
dataframes = pd.read_csv('challenge_dataset.txt', sep=',', header=None)
dataframes.columns = ["X_data", "Y_data"]
x_values = dataframes[['X_data']]
y_values = dataframes[['Y_data']]

# Train data with linear regression model
model = linear_model.LinearRegression()
model.fit(x_values, y_values)


# functio for calculating residuals
def calculate_residual(actual, predicted):
    res = (np.array(actual)-np.array(predicted)**2)
    return res


# Evaluating fit model
residuals = calculate_residual(y_values, model.predict(x_values))
print("Mean squared error: ", np.mean(residuals))
print("Coefficient of determination: ", model.score(x_values, y_values))
# Plotting on matplotlib
plt.figure(1)
plt.scatter(x_values, y_values, color="blue")
plt.savefig("Scatter.png")
plt.figure(2)
plt.plot(x_values, model.predict(x_values), color="black")
plt.figure(3)
plt.plot(residuals)
plt.savefig("residual.png")
plt.show()
