import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('FMLA1Q1Data_train.csv', header = None) # Loading train dataset
df_test = pd.read_csv('FMLA1Q1Data_test.csv', header = None) # Loading test dataset

bias = np.ones(len(df_train))
x_train = pd.DataFrame({'bias':bias, "x1":df_train.iloc[:,0], "x2":df_train.iloc[:,1]})
y_train = df_train.iloc[:,2].values

x_test = pd.DataFrame({'bias':np.ones(len(df_test)), "x1":df_test.iloc[:,0], "x2":df_test.iloc[:,1]})
y_test = df_test.iloc[:,2].values

"Finding type of kernal to use"

corr_all = np.corrcoef(df_train.values.T)
print("\nCorrelation Matrix among features and output\n\n",corr_all)

data = df_train.rename(columns={0: 'x1', 1: 'x2', 2: 'y'})

fig, ax1 = plt.subplots(1, 1)
sns.pairplot(data)
plt.suptitle("Relationship amongst features and output")

"""Strong Non linear relationship is observed between 'x1' and y and also 'x2 and'y',
Thus, we can use a polynomial kernel to capture non linearity"""

"To find ideal degree for our polynomial kernel, we can perform cross validation"  

class kernelregression:

    def __init__(self,X,y): # Initializing Parameters
        self.X = X
        self.y = y

    def poly_kernel (self, x_train, y_train, degree):
        k_train = ((x_train @ x_train.T) + 1) ** degree  # Kernel matrix is obtained by (XX^T + 1)^degree.
        return np.linalg.pinv(k_train) @ y_train         # alpha is obtained by K^(-1).y

    def degree_cv(self, folds, degree): # Method to validate degree of polynomila kernel
       
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

          alpha = self.poly_kernel(x_train, y_train, degree)
          k_val = ((x_train @ x_val.T) + 1)** degree
          y_pred = k_val.T @ alpha
          error = np.mean((y_val - y_pred)**2)
          fold_errors.append(error)
       return np.mean(fold_errors)

model = kernelregression(x_train,y_train)

cv_errors = []
for degree in range(1,11):
    error = model.degree_cv(5,degree)
    cv_errors.append(error) # List with validation errors for different values of degrees. 
best_degree = range(1,11)[np.argmin(np.round(cv_errors,2))]

print("\nMinimum validation error is:",min(cv_errors),", which is obtained from the polynimal kernel of degree:",best_degree)

"Testing kernel regression on Test data"

k_test = (x_train @ x_test.T + 1)**best_degree
alpha = model.poly_kernel(x_train,y_train,best_degree)
y_pred = k_test.T @ alpha # Predictions on Test data.
error = np.mean((y_test - y_pred)**2)

print("\nError obtained on the Test Data using the polynomial kernel with degree",best_degree,"is:",error,"\n")

ax1.plot(range(1, 11), cv_errors)
ax1.set_title("Validation Error vs Degree of polynomial kernel")
ax1.set_xlabel("Degree of Polynomial kernel")
ax1.set_ylabel("Validation Error")
ax1.grid(True)

plt.tight_layout()
plt.show()