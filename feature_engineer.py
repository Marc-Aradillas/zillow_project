import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pydataset import data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE # feature selection objects
from sklearn.linear_model import LinearRegression


# constants

lm = LinearRegression()

mms = MinMaxScaler()

# example on how to fit_transform after dropping cols.
# to_scale = df.drop(columns=['tip']).columns

# df[to_scale] = mms.fit_transform(df[to_scale])

# I could create a function to do scaling and operate in a stepping method.


# ________________________________ Select kBest function __________________________________

def select_kbest(X, y, k):

    # Initialize SelectKBest with the f_regression scoring function
    selector = SelectKBest(score_func=f_regression, k=k)
    
    # Fit the selector to the data
    selector.fit(X, y)
    
    # Get the indices of the top k selected features
    top_feature_indices = selector.get_support(indices=True)
    
    # Get the feature names based on the selected indices
    selected_features = pd.DataFrame(X.columns[top_feature_indices].tolist())

    # Remove the row and index from dataframe to display list
    selected_features_display = selected_features.style.hide_index().hide_columns()
    
    return selected_features_display



# ________________________________ RFE function ____________________________________

def rfe_(X, y, n):

    # Initialize with the LinearRegression estimator and n number of features
    rank = RFE(lm, n_features_to_select=n)
    
    # Fit the data
    rank.fit(X, y)
    
    # Get the indices of the top n ranked features
    top_feature_indices = rank.get_support(indices=True)
    
    # Get the feature names based on the selected indices
    ranked_features = pd.DataFrame(X.columns[top_feature_indices].tolist())

    #hiding index and columns
    ranked_features_display = ranked_features.style.hide_index().hide_columns()
    
    return ranked_features_display