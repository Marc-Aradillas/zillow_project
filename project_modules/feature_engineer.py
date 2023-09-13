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



import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from wrangle import acquire_zillow, clean_zillow, wrangle_zillow, train_val_test, split_and_scale_data, xy_split


def select_k_features(data, target_col='home_value', k=5):
    # Wrangle your data
    df = wrangle_zillow()

    # Drop categorical features (assuming regression model)
    df = df.drop(columns=['property_county_landuse_code', 'property_zoning_desc', 'n-prop_type', 'n-av_room_size', 'state'])

    # Split data
    train, val, test = train_val_test(df)

    # Split scale data
    X_train, y_train = xy_split(train, target_col)
    X_val, y_val = xy_split(val, target_col)
    X_test, y_test = xy_split(test, target_col)

    # Create a SelectKBest instance with k specified
    k_best = SelectKBest(score_func=f_regression, k=k)

    # Fit and transform your training data
    X_train_selected = k_best.fit_transform(X_train, y_train)

    # Get the indices of the selected features
    selected_indices = k_best.get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = X_train.columns[selected_indices]

    return selected_feature_names

# Example usage:
# selected_features = wrangle_and_select_features(data)
# print(selected_features)



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