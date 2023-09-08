import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from env import get_connection




# Constants
filename = 'zillow_data.csv'

# Acquire data.
# ----------------------ACQUIRE FUNCTION---------------------------------
def acquire_zillow():

    if os.path.isfile(filename):
        
        return pd.read_csv(filename)
        
    else: 

        query = '''
        SELECT p17.parcelid, bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, fullbathcnt, latitude, longitude, lotsizesquarefeet, regionidcounty,
        regionidzip, roomcnt, yearbuilt, taxvaluedollarcnt, censustractandblock, fips
        FROM properties_2017 AS p17
        LEFT JOIN predictions_2017 AS pr17 ON p17.parcelid = pr17.parcelid
        LEFT JOIN propertylandusetype AS plu ON p17.propertylandusetypeid = plu.propertylandusetypeid
        WHERE plu.propertylandusetypeid = 261 AND YEAR(pr17.transactiondate) = 2017; -- 'Single Family Residential' and transactions in 2017
        '''

        url = get_connection('zillow')
                
        df = pd.read_sql(query, url)

        # # save to csv
        # df.to_csv(filename,index=False)

        return df 


# Prep data.
# ---------------------------CLEAN FUNCTION----------------------------
def clean_zillow(df):

    """
    Cleans the Zillow data.
    
    Args:
    - df (DataFrame): Raw Zillow data.
    
    Returns:
    - cleaned_df (DataFrame): Cleaned Zillow data.
    """

    # Drop rows with duplicates
    df = df[~df.duplicated(keep='first')]
    
    # Drop rows with missing/null values in any column
    df = df.dropna()

    # Rename columns
    df = df.rename(columns={'bedroomcnt': 'bedrooms', 'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet': 'area', 'taxvaluedollarcnt': 'home_value',
                            'yearbuilt': 'year_built', 'fullbathcnt' : 'full_bath_cnt',
                            'lotsizesquarefeet' : 'lot_area', 'regionidcounty' : 'region_id_county',
                            'roomcnt' : 'room_cnt', 'censustractandblock' : 'census_tract_and_block'})

    # Convert selected columns to integer type
    int_columns = ['fips', 'year_built', 'home_value', 'area', 'bedrooms', 'full_bath_cnt', 'room_cnt',
                   'region_id_county', 'regionidzip', 'lot_area']
    df[int_columns] = df[int_columns].astype(int)

    fips_to_state = {
        6037: 'California',
        6059: 'California',
        6111: 'California',
        # Add more mappings for other states as needed
        }

    # Mapped county names to fips code
    fips_to_county = {
        6037: 'Los Angeles County',
        6059: 'Orange County',
        6111: 'Ventura County',
        # Add more mappings for other counties as needed
        }
    
    # Use the map method to create new 'county' and 'state' columns based on 'fips' column
    df['state'] = df['fips'].map(fips_to_state)
    df['county'] = df['fips'].map(fips_to_county)
    
    return df

# wrapping Acquire and Prep functions into one.
# ---------------------------- WRANGLE ZILLOW -------------------------------------
def wrangle_zillow():
    """
    Wrangles Zillow data by acquiring and cleaning it.
    
    Returns:
    - df (DataFrame): Wrangled Zillow data.
    """
    # Acquire data
    df = acquire_zillow()
    
    # Clean data
    df = clean_zillow(df)

    df.to_csv(filename, index=False)

    return df

# ------------------------ SPLIT FUNCTION -------------------------

def train_val_test(df, seed = 42):
    """
    splits cleaned df into train, validate, and split
    
    Returns:
    - train, validate, split subset of df (dataframe): Splitted Wrangles Zillow Data
    """
    
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)
    
    return train, val, test

# ------------------------ XY SPLIT FUNCTION ----------------------

# edit for zillow data 9/7/23 ; now its reusable
def xysplit(df, col):
    X = df.drop(columns=[col])
    y = df[col]
    return X, y

# examples:
# X_train, y_train = wrangle.xysplit(train, 'Species_setosa')
# X_val, y_val = wrangle.xysplit(val, 'Species_setosa')

# ------------------------ SCALE DATA FUNCTION --------------------
def scale_data(train, val, test, to_scale):
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
    validate_scaled[to_scale] = scaler.transform(val[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled