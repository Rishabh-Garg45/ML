import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import random

# Lodaing Dataset
data = load_dataset("mnist")
X_train= np.array(data['train']['image'])
y_train = np.array(data['train']['label'])
np.random.seed(42)  

print("\nSize of training data:", len(y_train))
print("\nResolution of training images:", np.shape(X_train[1]))

# Converting feature matrices to row vectors
flattened =np.shape(X_train[1])[0]*np.shape(X_train[1])[1]
print("\nOriginal number of Dimensions in training data:",flattened)

# Sampling 1000 datapoints for all classes
id = []
for label in range(10):
    class_id = []
    for j in range(len(y_train)):
        if y_train[j] == label:
            class_id.append(j)
    
    sampled_idx = random.sample(class_id, 100)
    id.extend(sampled_idx)

X_train = [X_train[i] for i in id]
y_train = [y_train[i] for i in id]

X_train = [np.array(i).reshape(flattened) for i in X_train]
y_train = np.array(y_train) 

print("\nSize of sampled Data:", len(X_train),"\n")

#PCA from Scratch
def pca_scratch(X):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    cov = np.cov(X_centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(-eigenvalues)   # Descending order of Eigenvalues 
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return mean, eigenvectors, eigenvalues

mean, eigenvectors, eigenvalues = pca_scratch(X_train)

#Q1.) Part a - Plotting first 20 PC's
plt.figure(figsize=(12, 12))
for i in range(20):
    plt.subplot(4, 5 , i+1)
    plt.imshow(eigenvectors[:, i].reshape(28, 28), cmap='gray')
    plt.title(f'Principal Component {i+1}')
plt.suptitle("Visualizing first 20 PC's")
plt.tight_layout(pad = 5)

explained_var = eigenvalues / np.sum(eigenvalues)

for i in range(20):
    print(f'Explained variance of Principal Component {i+1}: {explained_var[i]:10f}')
print('\n')

cu_var = 0
for i in range(20):
    cu_var = cu_var + explained_var[i]
    print(f'Cumulative variance of first {i+1} Principal Components: {cu_var:5f}')

print(f"\n{np.round(cu_var*100,2)} Percentage of Variance is captured by first 20 Principal components\n")
plt.show()

#Q2.) part b - Reconstruction

def reconstruct(X, mean, eigenvectors, d):
    
    # Project data onto the first 'd' principal components
    X_centered = X - mean
    X_proj = np.dot(X_centered, eigenvectors[:, :d])
    
    X_reconstruct = np.dot(X_proj, eigenvectors[:, :d].T) + mean
    return X_reconstruct

dimensions = [5, 10, 20, 50, 100, 150, 200, 300, flattened]  # different values of d
plt.figure(figsize=(15, 12))

#Sample a single digit for visualization
orig_image = X_train[750].reshape(28, 28)

#Plot the original image
plt.subplot(3, 4, 1)
plt.imshow(orig_image, cmap='gray')
plt.title(f'Original Image')
plt.axis('off')

#Reconstructing and visualizing using different numbers of principal components
for i, d in enumerate(dimensions, start=2):
    reconstructed_data = reconstruct(X_train, mean, eigenvectors, d)
    reconstructed_image = reconstructed_data[750].reshape(28, 28)
    
    plt.subplot(3, 4, i)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Reconstructed d={d}')
    plt.axis('off')

plt.suptitle("Recontructing using different number of dimensions")
plt.tight_layout(pad = 5)
plt.show()

#Analyzing explained variance for selecting d
cu_explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
plt.figure(figsize=(8, 5))
plt.plot(cu_explained_variance, marker='.')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.show()

#Print explained variance for key dimensions
for d in [5, 10, 20, 50, 100, 150, 200, 300]:
    print(f"d={d}: Cumulative Explained Variance = {cu_explained_variance[d-1]:.2f}")

print("\nConclusion - The given dataset's dimensionality can be reduced from 784 to 150 because with just 150 PC's, more than 95 percent of variance is captured\n")