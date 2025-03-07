import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from scipy.linalg import svd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import KFold


def init():
    # Fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframe)
    X = adult.data.features
    # print(X)
    y = adult.data.targets
    y = y.loc[X.index].squeeze()
    y = y.map({'<=50K': 0, '>50K': 1})
    ###

    # metadata
    #print(adult.metadata)

    # variable information
    #print(adult.variables)

    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    return (X, y)

def part_one(X, y):
    #######################################
    #                PART 1
    #######################################
    # Convert "?" to NaN
    X.replace("?", np.nan, inplace=True)

    # Remove rows with missing values
    X.dropna(inplace=True)

    # Ensure y only includes rows that are still in X
    y = y.loc[X.index]

    # Remove noise
    X.drop(columns=['fnlwgt'], inplace=True)

    # Convert categorical features into numerical (one-hot encoding for non-ordinal categories)
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def part_two(X):
    #######################################
    #                PART 2
    #######################################

    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    #print("Feature means before standardization:", feature_means)
    #print("std before standardization:", feature_stds)

    # standardize set
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # standardize, then report mean and std (should be 0, 1)
    feature_means = np.mean(X_standardized, axis=0)
    feature_stds = np.std(X_standardized, axis=0)
    #print("Feature means after standardization:", feature_means)
    #print("std after standardization:", feature_stds)

    # K-means clustering
    k_results = []
    k_values = range(1, 21)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_standardized)
        k_results.append(kmeans.inertia_)

    # Best k
    best_k = k_values[np.argmin(np.gradient(np.gradient(k_results)))]
    #print("Best k", best_k)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = svd(X_standardized, full_matrices=False)
    #print("Top singular values:", S[:5])
    top_feature_indices = np.argsort(np.abs(Vt[0]))[::-1][:5]
    top_features = X.columns[top_feature_indices]
    #print("Top 5 contributing features:", top_features.tolist())

    # Construct correlation matrix
    correlation_matrix = np.corrcoef(X_standardized.T)
    #print("Correlation Matrix:\n", correlation_matrix)

    high_correlation_pairs = []
    threshold = 0.5
    num_features = correlation_matrix.shape[0]

    for i in range(num_features):
        for j in range(i + 1, num_features):  # no diagonals
            if abs(correlation_matrix[i, j]) > threshold:
                feature_i = X.columns[i]
                feature_j = X.columns[j]
                high_correlation_pairs.append((feature_i, feature_j, correlation_matrix[i, j]))

    #print("Highly correlated features (index pairs and correlation values):", high_correlation_pairs)
    return X


def part_three(X, y):
    #########################################
    #                PART 3
    #########################################
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Basic Linear Model
    model = LinearRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_standardized, y, cv=kf, scoring='neg_root_mean_squared_error')
    print("Basic Linear Model RMSE:", -np.mean(scores))

    # Save model params for each fold
    model_params = []
    for train_idx, val_idx in kf.split(X_standardized):
        model.fit(X_standardized[train_idx], y.to_numpy()[train_idx])
        model_params.append(model.coef_)

    np.save("linear_model_params.npy", np.array(model_params))

    # Feature Engineering: Add k-means cluster labels as a feature
    best_k = 5
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_standardized)
    X_clustered = np.column_stack((X_standardized, cluster_labels))

    # Regularized Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    scores = cross_val_score(ridge_model, X_clustered, y, cv=kf, scoring='neg_root_mean_squared_error')
    print("Regularized Model RMSE:", -np.mean(scores))

    # Tune Regularization Parameter
    alphas = [0.1, 1, 10, 100]
    best_alpha = None
    best_rmse = float("inf")
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_clustered, y, cv=kf, scoring='neg_root_mean_squared_error')
        avg_rmse = -np.mean(scores)
        print(f"Alpha {alpha}, RMSE: {avg_rmse}")
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_alpha = alpha

    print("Best Regularization Alpha:", best_alpha)

    # Train-Test Split and Evaluate Best Ridge Model
    X_train, X_test, y_train, y_test = train_test_split(X_clustered, y, test_size=0.2, random_state=42)
    best_ridge = Ridge(alpha=best_alpha)
    best_ridge.fit(X_train, y_train)
    y_pred = best_ridge.predict(X_test)

    # Fix RMSE computation
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Test Set RMSE with Best Ridge Model:", test_rmse)

    # Save model parameters
    np.save("ridge_model_params.npy", best_ridge.coef_)

    return X

def main():
    X,y = init()
    X, y = part_one(X, y)
    X = part_two(X)
    X = part_three(X, y)
    print("Program ending, goodbye!")


if __name__ == "__main__":
    main()
