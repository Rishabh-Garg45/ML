import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FMLA1Q1Data_train.csv', header = None) # Loading Dataset

print(df.head(10)) # view of the data
print(df.describe()) # summary of the data


"""
   Addition of the bias term to dataset for Linear Regression. 
   Least squares solution needs bias term to adjust output independent of the input features.
"""

bias = np.ones(len(df))
x_train = pd.DataFrame({'bias':bias, "x1":df.iloc[:,0], "x2":df.iloc[:,1]})
y_train = df.iloc[:,2].values

print(x_train.head())

" Analytical Solution class "

class leastsquares_analytical:
  def __init__(self, X, y): # Initializing x_train and y_train
    self.X = X # n x d dimensional matrix having n datapoints with d features each
    self.y = y # n x 1 dimensional vector having output values for each datapoint.

  def compute_wml(self):  # Method to compute weights.

    # Analytical solution is given by formula w_ml = (XX^T)^(-1)Xy

    self.w_ml = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y) 

    return self.w_ml

model = leastsquares_analytical(x_train, y_train) # Creating Object of the analytical solution class

w_ml = model.compute_wml() # Computing weights

print("Weights using Analytical Solution:",w_ml)

y_pred = x_train.dot(w_ml) # Predictions using computed weights: y_pred = X^T.w_ml

mse = np.sum((y_train - y_pred)**2)/len(y_train) # Mean Squared error of training set of the model

print("Mean squared error:", mse)