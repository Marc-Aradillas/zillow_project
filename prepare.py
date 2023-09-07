# imported libs for scaling
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

#custom import
import wrangle

# -----------------Train-Validate-Test-------------------------------

seed = 42

# function to subset data
def train_val_test(df, seed = 42):

    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)

    return train, val, test



# - - - - - - - - - - - Scale Data - - - - - - - - - - - - - - - -
# Define a function for scaling the data
def scale_data(train, val, test, scaler):

    # # Filter data
    # train = train[(train['tax_value'] < 3_000_000) & (train['tax_amount'] < 3_000_000)]
    # val = val[(val['tax_value'] < 3_000_000) & (val['tax_amount'] < 3_000_000)]
    # test = test[(test['tax_value'] < 3_000_000) & (test['tax_amount'] < 3_000_000)]

    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    columns_to_scale = ['bedrooms', 'bathrooms', 'area', 'year_built', 'tax_value']
    
    # Fit the scaler on the training data for all of the columns
    scaler.fit(train[columns_to_scale])
    
    # Transform the data for each split
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(val[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    scaled_col = [train_scaled, validate_scaled, test_scaled]
    
    return train_scaled, validate_scaled, test_scaled


#------------- Instructor's Function -----------------------------
"""
to_scale = ['bedrooms','bathrooms','area','year_built','tax_value']

def scale_data(train, validate, test, to_scale):
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled

# used this way:
train_scaled, validate_scaled, test_scaled = scale_data(train, val, test, to_scale)
"""

# --------------------Visualization functions----------------------

def visualize_compare(scaled_col, df, original):
    plt.figure(figsize=(11, 7))

    plt.subplot(121)
    sns.histplot(data=df, x=original, bins=40)
    plt.title(f'Distribution original')
    
    plt.subplot(122)
    sns.histplot(data=df, x=scaled_col, bins=40)
    plt.title(f'Distribution scaled')

    plt.tight_layout()
    plt.show()


def visualize_all_columns(scaled_df, df):
    for col in scaled_df.columns:
        visualize_compare(scaled_df[col], df, col)

