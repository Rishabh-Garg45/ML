import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FMLA1Q1Data_train.csv', header = None) # Loading Dataset

bias = np.ones(len(df))
x_train = pd.DataFrame({'bias':bias, "x1":df.iloc[:,0], "x2":df.iloc[:,1]})
y_train = df.iloc[:,2].values

class StochasticGD:

    "Applying GD with batch size of 100."

    def __init__(self, X, y, wml): # Initialization of Parameters
        self.X = X
        self.y = y
        self.wml = wml

    def compute_weights(self): # Function to get weights
      
      self.w = np.zeros(self.X.shape[1])
      self.w_track = []
      indices = np.random.permutation(1000).tolist() # Random list from 1 to 1000 for shuffling

      for i in range(1,10000):

        if not indices: # Until indicies are not exhausted
          indices = np.random.permutation(1000).tolist()
        
        selected_indices = indices[:100] # Selecting first 100 random indicies from list
        del indices[:100] # removing those 100 from list

        xi = self.X.iloc[selected_indices] # x_train have datapoints with those random indicies
        yi = self.y[selected_indices] # y_train have corresponding output values

        self.w_new = self.w - (1/(i+10))*(2 / xi.shape[0]) * xi.T.dot(xi.dot(self.w) - yi)
        self.norm = np.linalg.norm(self.w_new - self.wml)

        self.w_track.append(self.norm)

        if np.round(np.linalg.norm(self.w_new - self.w),3) == 0: # Convergence Condition-> When the gradient is Zero, Zero weight error
           print(f"\nStochastic_GD converged after {i} iterations\n")
           return self.w_new, i, self.w_track
        
        self.w = self.w_new
      return self.w, i, self.w_track

w_ml  = [9.89400832, 1.76570568, 3.5215898] # w_ml reported by Analytical solution (rounded off to 3 decimals)
model = StochasticGD(x_train, y_train, w_ml) # Creating object of stochastic gradient descent class

w_sgd, actual_iter, w_track= model.compute_weights() # Computing weights

"As expected, Stochastic GD requires more iteration to converge as compared to Gradient Descent"

print("Weights computed using Stochastic Gradient Descent:\n\n",w_sgd)

y_pred = x_train.dot(w_sgd) # Predictions using computed weights

mse = np.sum((y_train - y_pred)**2)/len(y_train) # Mean Squared error using Stochastic Gradient Descent algorithm.

print("\nMean Squared error:",mse)

plt.plot(range(1,actual_iter + 1), w_track)
plt.xlim(-7, 500)
plt.ylim(-0.15, 2)
plt.xlabel("Iterations")
plt.ylabel("L2 Norm of weight differences")
plt.title("L2 Norm of weight differences vs Iterations")
plt.show()