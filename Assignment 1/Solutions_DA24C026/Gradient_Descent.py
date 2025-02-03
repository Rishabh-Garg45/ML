import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FMLA1Q1Data_train.csv', header = None) # Loading Dataset

bias = np.ones(len(df))
x_train = pd.DataFrame({'bias':bias, "x1":df.iloc[:,0], "x2":df.iloc[:,1]})
y_train = df.iloc[:,2].values

class gradientDescent:
  
  def __init__(self, X, y, wml): # Initialization of Parameters
    self.X = X
    self.y = y
    self.wml = wml

  def compute_weights(self): # Function to compute weights using iterative approach 
    
    " GD update rule -> w_new = w_old - a * d/dw (1/n) * Σ(w^t.x - y)^2, a is the learning rate "

    self.w = np.zeros(self.X.shape[1]) # Initializing weights.
    self.w_track = [] # storing norm of weight error (w_new - w_ml)
    
    for i in range(1,10000):
      
      self.w_new = self.w - (1/(i+10))*(2 / self.X.shape[0]) * self.X.T.dot(self.X.dot(self.w) - self.y) # 1/i is dynamic learning rate that satisfies LR properties.
      self.norm = np.linalg.norm(self.w_new - self.wml) # L2 norm of difference between w_new and w_ml
      self.w_track.append(self.norm)

      if not np.round(np.linalg.norm(self.w_new - self.w),3): # Convergence Condition-> When the gradient is Zero (no change in weights)
        print(f"\nGD converged after {i} iterations\n")
        return self.w_new, i, self.w_track
      
      self.w = self.w_new
    return self.w, i, self.w_track

w_ml  = [9.89400832, 1.76570568, 3.5215898] # w_ml reported by Analytical solution (rounded off to 3 decimals)
model = gradientDescent(x_train, y_train, w_ml) # Creating object of gradient descent class

w_gd, actual_iter, w_track= model.compute_weights() # Computing weights

print("Weights computed using Gradient Descent:\n\n",w_gd)

y_pred = x_train.dot(w_gd) # Predictions using computed weights

mse = np.sum((y_train - y_pred)**2)/len(y_train) # Mean Squared error using Gradient Descent algorithm.

print("\nMean Squared error:",mse,"\n")

" Plotting L2 norm of weight errors as function of t "

plt.plot(range(1,actual_iter + 1), w_track)
plt.xlabel("Iterations")
plt.ylabel("L2 norm of weight differences")
plt.title("L2 norm of weight differences vs Iterations")
plt.show()