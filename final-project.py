import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.linalg import svd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframe)
X = adult.data.features
#print(X)
y = adult.data.targets
###

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

#######################################
#                PART 1
#######################################
# Convert "?" to NaN
X.replace("?", np.nan, inplace=True)

# remove rows with incomplete values
X.dropna(inplace=True)

# remove noise
X.drop(columns=['fnlwgt'], inplace=True)

# Convert categorical features into numerical (one-hot encoding for non-ordinal categories)
X = pd.get_dummies(X, drop_first=True)


#print("Cleaned Up Data")
#print(X.columns)
#print(X.shape)


#######################################
#                PART 2
#######################################

feature_means = np.mean(X, axis=0)
feature_stds = np.std(X, axis=0)
print("Feature means before standardization:", feature_means)
print("std before standardization:", feature_stds)

# standardize set
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# standardize, then report mean and std (should be 0, 1)
feature_means = np.mean(X_standardized, axis=0)
feature_stds = np.std(X_standardized, axis=0)
print("Feature means after standardization:", feature_means)
print("std after standardization:", feature_stds)

# K-means clustering
k_results = []
k_values = range(1, 21)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_standardized)
    k_results.append(kmeans.inertia_)

# Best k
best_k = k_values[np.argmin(np.gradient(np.gradient(k_results)))]
print("Best k", best_k)

# Perform Singular Value Decomposition (SVD)
U, S, Vt = svd(X_standardized, full_matrices=False)
print("Top singular values:", S[:5])
top_feature_indices = np.argsort(np.abs(Vt[0]))[::-1][:5]
top_features = X.columns[top_feature_indices]
print("Top 5 contributing features:", top_features.tolist())

# Construct correlation matrix
correlation_matrix = np.corrcoef(X_standardized.T)
print("Correlation Matrix:\n", correlation_matrix)

high_correlation_pairs = []
threshold = 0.5
num_features = correlation_matrix.shape[0]

for i in range(num_features):
    for j in range(i + 1, num_features):  # no diagonals
        if abs(correlation_matrix[i, j]) > threshold:
            feature_i = X.columns[i]
            feature_j = X.columns[j]
            high_correlation_pairs.append((feature_i, feature_j, correlation_matrix[i, j]))

print("Highly correlated features (index pairs and correlation values):", high_correlation_pairs)

print("Program ending, goodbye!")
