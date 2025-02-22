import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# Fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframe)
X = adult.data.features
#print(X)
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

# Convert "?" to NaN
X.replace("?", np.nan, inplace=True)

# remove rows with incomplete values
X.dropna(inplace=True)

# remove noise
X.drop(columns=['fnlwgt'], inplace=True)

# Convert categorical features into numerical (one-hot encoding for non-ordinal categories)
X = pd.get_dummies(X, drop_first=True)


print("Cleaned Up Data")
print(X.columns)
print(X.shape)
print("Program ending, goodbye!")
