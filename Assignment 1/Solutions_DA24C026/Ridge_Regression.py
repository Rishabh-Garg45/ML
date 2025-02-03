import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('FMLA1Q1Data_train.csv', header = None) # Loading train dataset
df_test = pd.read_csv('FMLA1Q1Data_test.csv', header = None) # Loading test dataset

bias = np.ones(len(df_train))
x_train = pd.DataFrame({'bias':bias, "x1":df_train.iloc[:,0], "x2":df_train.iloc[:,1]})
y_train = df_train.iloc[:,2].values

x_test = pd.DataFrame({'bias':np.ones(len(df_test)), "x1":df_test.iloc[:,0], "x2":df_test.iloc[:,1]})
y_test = df_test.iloc[:,2].values

class RidgeRegression:
    
    " Ridge GD update rule -> w_new = w_old - a * d/dw ((1/n) * Σ(w^t.x - y)^2 +(λ * Σ w^2)), 'a' is the learning rate. "

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def gradientdescent(self, x_train, y_train, lambda_): # Method for computing w_r for each lambda
       
       w = np.zeros(x_train.shape[1])
       for i in range(1,10000):
            w_new = w - (0.5/len(x_train))*((x_train.T.dot(x_train.dot(w) - y_train)) + (lambda_ * w))
            if np.round(np.linalg.norm(w_new - w),3) == 0: # Convergence Condition-> When the gradient is Zero, Zero weight error
                return w_new
            w = w_new
       return w
    
    def lambda_cval(self, folds, lambda_): # Cross validating different values of Lambda
       
       fold_errors = []
       indices = np.arange(1000)
       val_size = len(self.X)/folds
       
       for i in range(folds):
       
          val_index = indices[int(i * val_size) : int((i+1) * val_size)]
          train_index = [i for i in indices if i not in val_index]
          x_train = self.X.iloc[train_index] 
          y_train = self.y[train_index] 
          x_val  = self.X.iloc[val_index]
          y_val = self.y[val_index]
          w_r = self.gradientdescent(x_train,y_train,lambda_)
          y_pred = x_val.dot(w_r)
          error = np.mean((y_pred - y_val)**2)
          fold_errors.append(error)
       
       return np.mean(fold_errors)

model = RidgeRegression(x_train, y_train)

c_val_errors = []
for lambda_ in np.arange(1,15.5,0.5):
    c_val_errors.append(model.lambda_cval(folds = 5,lambda_ = lambda_)) # Finding validation set errors for different lambda values

min_error = np.argmin(c_val_errors)
best_lambda = np.arange(1,15.5,0.5)[min_error]

print("\nBest Lambda reported for Ridge Regression is :", best_lambda,"\nWith minimum validation error:",min(c_val_errors))

" Computing w_r for best Lambda"

w_r = model.gradientdescent(x_train,y_train,3.5)
print("\nWeight obtained using Ridge Regression are: \n", w_r)

"Test Error using Analytical solution"

w_ml  = [9.89400832, 1.76570568, 3.5215898] # Analytical solution weights
test_error = x_test.dot(w_ml)
print("\nTest Error for weights obtained using Analytical solution is :", np.mean((test_error-y_test)**2))

"Test error using ridge regression"

y_pred_r = x_test.dot(w_r)
print("Test Error for weights obtained using Ridge Regression is :", np.mean((y_pred_r-y_test)**2))

"Test error using ridge regression solution is observed to be lesser than analytical solution"
"Validation set error vs Lambda plot"

plt.plot(np.arange(1,15.5,0.5), c_val_errors)
plt.xticks(np.arange(1,15.5,0.5))
plt.xlabel('Lambda values')
plt.ylabel('Validation Error')
plt.grid(True)
plt.title('Cross Validation Errors of Lambda values')
plt.show()