import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from env import get_connection




# Constant (to generate filename for csv)
filename = 'zillow_data.csv'

# Acquire data.
# ----------------------ACQUIRE FUNCTION---------------------------------
def acquire_zillow():

    if os.path.isfile(filename):
        
        return pd.read_csv(filename)
        
    else: 

        query = '''
                SELECT p17.parcelid, latitude, longitude, lotsizesquarefeet, regionidcounty, regionidcity, regionidzip, propertycountylandusecode, propertyzoningdesc,bathroomcnt, bedroomcnt, calculatedbathnbr, 
                fullbathcnt, calculatedfinishedsquarefeet, basementsqft, finishedsquarefeet12, finishedsquarefeet15, finishedsquarefeet50, fips, roomcnt, numberofstories, yearbuilt, plu.propertylandusetypeid, 
                airconditioningtypeid, buildingqualitytypeid, heatingorsystemtypeid, architecturalstyletypeid, buildingclasstypeid, decktypeid, typeconstructiontypeid, unitcnt, garagecarcnt, garagetotalsqft, poolcnt, 
                poolsizesum, pooltypeid10, pooltypeid2, pooltypeid7, fireplacecnt, fireplaceflag, hashottuborspa, yardbuildingsqft17, yardbuildingsqft26, threequarterbathnbr, taxvaluedollarcnt, taxdelinquencyflag, 
                taxdelinquencyyear, rawcensustractandblock, censustractandblock
                
                FROM properties_2017 AS p17
                
                LEFT JOIN predictions_2017 AS pr17 ON p17.parcelid = pr17.parcelid
                
                LEFT JOIN propertylandusetype AS plu ON p17.propertylandusetypeid = plu.propertylandusetypeid
                
                WHERE plu.propertylandusetypeid = 261 AND YEAR(pr17.transactiondate) = 2017; -- 'Single Family Residential' and transactions in 2017017; -- 'Single Family Residential' and transactions in 2017
                '''

        url = get_connection('zillow')
                
        df = pd.read_sql(query, url)

        # save to csv
        df.to_csv(filename,index=False)

        return df 


# ---------------------------CLEAN FUNCTION--------------------------
# Prep data.
def clean_zillow(df):
    """
    Cleans the Zillow data.
    
    Args:
    - df (DataFrame): Raw Zillow data.
    
    Returns:
    - df (DataFrame): Cleaned Zillow data.
    """
    # Pool features edits
    df.poolcnt.fillna(0, inplace=True)
    df['hashottuborspa'].fillna(0, inplace=True)
    df['hashottuborspa'].replace(True, 1, inplace=True)
    df['poolsizesum'] = df['poolsizesum'].fillna(df[df['poolcnt'] == 1]['poolsizesum'].median())
    df.loc[df.poolcnt==0, 'poolsizesum']=0
    df['pooltypeid2'].fillna(0, inplace=True)
    df['pooltypeid7'].fillna(0, inplace=True)
    df.drop('pooltypeid10', axis=1, inplace=True)

    # Fireplace feature edits
    df.loc[(df['fireplaceflag'] == True) & (df['fireplacecnt'].isnull()), 'fireplacecnt'] = 1
    df['fireplacecnt'].fillna(0, inplace=True)
    df.loc[(df['fireplacecnt'] >= 1.0) & (df['fireplaceflag'].isnull()), 'fireplaceflag'] = True
    df['fireplaceflag'].fillna(0, inplace=True)
    df['fireplaceflag'].replace(True, 1, inplace=True)

    # Garage feature edits
    df['garagecarcnt'].fillna(0, inplace=True)
    df['garagetotalsqft'].fillna(0, inplace=True)

    # Tax delinquency feature edits
    df['taxdelinquencyflag'].fillna(0, inplace=True)
    df['taxdelinquencyflag'].replace('Y', 1, inplace=True)
    df.drop('taxdelinquencyyear', axis=1, inplace=True)

    # Other features
    bathroommode = df['calculatedbathnbr'].value_counts().argmax() # should be 0
    df['calculatedbathnbr'] = df['calculatedbathnbr'].fillna(bathroommode)
    df['basementsqft'].fillna(0, inplace=True)
    df['yardbuildingsqft26'].fillna(0, inplace=True)
    df.drop('architecturalstyletypeid', axis=1, inplace=True)
    df.drop('typeconstructiontypeid', axis=1, inplace=True)
    df.drop('fullbathcnt', axis=1, inplace=True)
    df.drop('threequarterbathnbr', axis=1, inplace=True)
    df.drop('buildingclasstypeid', axis=1, inplace=True)
    df['decktypeid'].fillna(0, inplace=True)
    df['decktypeid'].replace(66.0, 1, inplace=True)
    df['calculatedfinishedsquarefeet'].fillna(df['calculatedfinishedsquarefeet'].mean(), inplace=True)
    df.loc[df['finishedsquarefeet15'].isnull(), 'finishedsquarefeet15'] = df['calculatedfinishedsquarefeet']
    df.loc[df['finishedsquarefeet12'].isnull(), 'finishedsquarefeet12'] = df['calculatedfinishedsquarefeet']
    df['numberofstories'].fillna(1, inplace=True)
    df.loc[df['numberofstories'] == 1.0, 'finishedsquarefeet50'] = df['calculatedfinishedsquarefeet']
    df['finishedsquarefeet50'].fillna(df['finishedsquarefeet50'].mean(), inplace=True)
    df['yardbuildingsqft17'].fillna(0, inplace=True)
    df['airconditioningtypeid'].fillna(5, inplace=True)
    df['heatingorsystemtypeid'].fillna(13, inplace=True)
    df['buildingqualitytypeid'].fillna(df['buildingqualitytypeid'].value_counts().idxmax(), inplace=True)
    df['unitcnt'].fillna(1, inplace=True)
    df['propertyzoningdesc'].fillna(df['propertyzoningdesc'].value_counts().idxmax(), inplace=True)
    df['lotsizesquarefeet'].fillna(df['lotsizesquarefeet'].value_counts().idxmax(), inplace=True)
    df.drop('censustractandblock', axis=1, inplace=True)
    df['taxvaluedollarcnt'].fillna(df['taxvaluedollarcnt'].mean(), inplace=True)
    df['yearbuilt'].fillna(df['yearbuilt'].value_counts().idxmax(), inplace=True)
    df.drop('regionidcity', axis=1, inplace=True)
    df['regionidzip'].fillna(df['regionidzip'].value_counts().idxmax(), inplace=True)


    # renaming all features as appropriate
    df = df.rename(columns={'parcelid': 'parcel_id', 'pooltypeid2': 'pool_type_id_2', 'bedroomcnt': 'bedrooms',
                            'bathroomcnt': 'bathrooms', 'heatingorsystemtypeid': 'heating_or_system_type_id',
                            'decktypeid': 'deck_type_id', 'unitcnt': 'unit_cnt', 'garagecarcnt': 'garage_car_cnt',
                            'calculatedfinishedsquarefeet': 'area', 'poolcnt': 'pool_cnt', 'taxvaluedollarcnt': 'home_value',
                            'garagetotalsqft': 'garage_total_sqft', 'yearbuilt': 'year_built', 'fullbathcnt': 'full_bath_cnt',
                            'lotsizesquarefeet': 'lot_area', 'regionidcounty': 'region_id_county', 'roomcnt': 'room_cnt',
                            'rawcensustractandblock': 'raw_census_tract_and_block', 'poolsizesum': 'pool_size_sum',
                            'pooltypeid7': 'pool_type_id_7', 'fireplacecnt': 'fire_place_cnt', 'fireplaceflag': 'fire_place_flag',
                            'hashottuborspa': 'has_hot_tub_or_spa', 'yardbuildingsqft17': 'patio_sqft',
                            'yardbuildingsqft26': 'storage_sqft', 'taxdelinquencyflag': 'tax_delinquency_flag',
                            'buildingqualitytypeid': 'building_quality_type_id', 'airconditioningtypeid': 'ac_type_id',
                            'numberofstories': 'num_stories', 'regionidzip': 'region_id_zip',
                            'propertycountylandusecode': 'property_county_landuse_code',
                            'propertyzoningdesc': 'property_zoning_desc', 'calculatedbathnbr': 'calc_bath_count',
                            'basementsqft': 'basement_sqft', 'finishedsquarefeet15': 'finished_sqft_15',
                            'finishedsquarefeet50': 'finishedsqft50', 'roomcnt': 'rooms',
                            'finishedsquarefeet12': 'finished_square_feet_12', 'finishedsqft50': 'finished_sqft_50',
                            'propertylandusetypeid': 'property_landuse_type_id'})

    # Additional Features created reference: https://www.kaggle.com/code/nikunjm88/creating-additional-features/notebook
    # living space
    df['n-life'] = 2023 - df['year_built']
    df['n-living_area_error'] = df['area'] / df['finished_square_feet_12']
    df['n-living_area_prop'] = df['area'] / df['lot_area']
    df['n-living_area_prop2'] = df['finished_square_feet_12'] / df['finished_sqft_15']

    # other space consideration and property bundle
    df['n-extra_space'] = df['lot_area'] - df['area']
    df['n-extra_space-2'] = df['finished_sqft_15'] - df['finished_square_feet_12']
    df['n-total_rooms'] = df['bathrooms'] * df['bedrooms']
    df['n-av_room_size'] = df['area'] / df['rooms']
    df['n-extra_rooms'] = df['rooms'] - df['n-total_rooms']
    df['n-gar_pool_ac'] = ((df['garage_car_cnt'] > 0) & (df['pool_type_id_7'] > 0) & (df['ac_type_id'] != 5)) * 1

    # latitude and longitude decimal palce error fix
    df["n-location"] = df["latitude"] + df["longitude"]
    df["n-location-2"] = df["latitude"] * df["longitude"]
    df["n-location-2round"] = df["n-location-2"].round(-4)
    df["n-latitude-round"] = df["latitude"].round(-4)
    df["n-longitude-round"] = df["longitude"].round(-4)

    # zip and county count mapping
    zip_count = df['region_id_zip'].value_counts().to_dict()
    df['n-zip_count'] = df['region_id_zip'].map(zip_count)
    region_count = df['region_id_county'].value_counts().to_dict()
    df['n-county_count'] = df['region_id_county'].map(region_count)

    # ac-heat/property typ consolidation
    df['n-ac_ind'] = (df['ac_type_id'] != 5) * 1
    df['n-heat_ind'] = (df['heating_or_system_type_id'] != 13) * 1
    df['n-prop_type'] = df.property_landuse_type_id.replace({31: "Mixed", 46: "Other", 47: "Mixed", 246: "Mixed",
                                                             247: "Mixed", 248: "Mixed", 260: "Home", 261: "Home",
                                                             262: "Home", 263: "Home", 264: "Home", 265: "Home",
                                                             266: "Home", 267: "Home", 268: "Home", 269: "Not Built",
                                                             270: "Home", 271: "Home", 273: "Home", 274: "Other",
                                                             275: "Home", 276: "Home", 279: "Home", 290: "Not Built",
                                                             291: "Not Built" })

    
    # Convert selected columns to integer type
    int_columns = ['fips', 'year_built', 'home_value', 'area', 'bedrooms', 'rooms',
                       'region_id_county', 'region_id_zip', 'lot_area']
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

    # One-hot encode the 'county' column
    df = pd.get_dummies(df, columns=['county'], prefix='', prefix_sep='')

    columns_to_convert = ['Los Angeles County', 'Orange County', 'Ventura County']
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    # Define bin edges and labels
    bin_edges = [0, 1000, 2000, float('inf')]
    bin_labels = ['Small', 'Medium', 'Large']
    
    # Create a new column 'property_size' with the categories
    df['property_size'] = pd.cut(df['area'], bins=bin_edges, labels=bin_labels)
    
    # Create dummy variables for 'property_size'
    dummies = pd.get_dummies(df['property_size'])
    
    # Concatenate the dummy variables with the original DataFrame
    df = pd.concat([df, dummies], axis=1)
    
    # Drop the 'property_size' column if needed
    df.drop('property_size', axis=1, inplace=True)
    
    # Convert the dummy variables to integers
    columns_to_convert = ['Small', 'Medium', 'Large']
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    return df

# ----------------------- WRANGLE ZILLOW ------------------------
# wrapping Acquire and Prep functions into one.
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

    return df


# ------------------------ SPLIT FUNCTION -------------------------
# train val test split function
def train_val_test(df, seed = 42):
    """
    splits cleaned df into train, validate, and split
    
    Returns:
    - train, validate, split subset of df (dataframe): Splitted Wrangled Zillow Data
    """
    # data is split into 70% train and from the 30%, 50% goes to each test and validate subsets.
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)

    
    return train, val, test


# ------------------------ SPLIT/SCALE FUNCTION -------------------------
# train val test split/scale function
def split_and_scale_data(df, seed=42):
    """
    Splits the data into train, validate, and test sets, and scales all numerical features.
    
    Parameters:
    - df (dataframe): The input dataframe.
    - seed (int): Random seed for reproducibility.

    Returns:
    - train_scaled, validate_scaled, test_scaled (dataframes): Scaled train, validate, and test sets.
    """

    # Split the data into train, validate, and test sets
    train, val, test = train_val_test(df, seed)

    # Extract the numerical features to scale (assuming all non-categorical columns are numerical)
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()

    # Scale all numerical features using Min-Max scaling
    scaler = MinMaxScaler()
    scaler.fit(train[numerical_features])
    train[numerical_features] = scaler.transform(train[numerical_features])
    val[numerical_features] = scaler.transform(val[numerical_features])
    test[numerical_features] = scaler.transform(test[numerical_features])

    return train, val, test


# ------------------------ XY SPLIT FUNCTION ----------------------
# xy_split function to create usable subsets; reusable.
def xy_split(df, col):
    X = df.drop(columns=[col])
    y = df[col]
    return X, y


# ------------------------ XY SPLIT TVT FUNCTION ----------------------
def scale_data(train, val, test, to_scale):
    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    #make the thing
    scaler = StandardScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(val[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled

