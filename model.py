import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wrangle import wrangle_zillow, train_val_test, xy_split
from evaluate import plot_residuals, regression_errors, baseline_mean_errors, better_than_baseline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from math import sqrt



# ------------------------ SCALE DATA FUNCTION --------------------
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

def eval_model(y_actual, y_hat):
    
    return sqrt(mean_squared_error(y_actual, y_hat))


def train_model(model, X_train, y_train, X_val, y_val):
    
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is {train_rmse:.2f}.\n')
    print(f'The validate RMSE is {val_rmse:.2f}.\n\n')
    
    return model


# ------------------------------ Train and eval function -------------------------------------

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Train a machine learning model and evaluate its performance on training and validation data.
    
    Args:
        model (object): The machine learning model to be trained.
        X_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        X_val (array-like): Validation feature data.
        y_val (array-like): Validation target data.
        
    Returns:
        object: The trained model.
        float: The training RMSE.
        float: The validation RMSE.
    """
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on training and validation data
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Evaluate the model's performance
    train_rmse = eval_model(y_train, train_preds)
    val_rmse = eval_model(y_val, val_preds)

    # Calculate R-squared (R2) for training and validation sets
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    # Print the results
    print(f"\n-------------------------------------")
    print(f'The train RMSE is {train_rmse:.2f}.\n')
    print(f"\n-------------------------------------")
    print(f'The validation RMSE is {val_rmse:.2f}.\n\n')
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    
    return model, train_rmse, val_rmse

# Example usage:
# trained_model, train_rmse, val_rmse = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)


# --------------------------------- Zillow plot and model train -----------------------\

def wrangle_zillow_and_train_model():
    # Wrangle the data
    df = wrangle_zillow()

    # Drop categorical features
    df = df.drop(columns=['property_county_landuse_code', 'property_zoning_desc', 'n-prop_type', 'n-av_room_size', 'state'])

    # Train-test split
    train, val, test = train_val_test(df)

    # Split data into X and y for train and val
    X_train, y_train = xy_split(train, 'home_value')
    X_val, y_val = xy_split(val, 'home_value')

    # Calculate baseline
    bl = y_train.median()

    # Create a DataFrame to work with
    preds = pd.DataFrame({'y_actual' : y_train,
                          'y_baseline': bl})

    # Calculate baseline residuals
    preds['y_baseline_residuals'] = bl - preds['y_actual']

    # Initialize and fit a linear regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Make predictions with the model
    preds['y_hat'] = lm.predict(X_train)

    # Calculate model residuals
    preds['y_hat_residuals'] = preds['y_hat'] - preds['y_actual']

    # Plot residuals
    plot_residuals(preds.y_actual, preds.y_hat)

    print(f"\n-------------------------------------")
    # Calculate regression errors
    SSE, ESS, TSS, MSE, RMSE = regression_errors(preds.y_actual, preds.y_hat)
    print(f"\nModel RMSE: {RMSE:.2f}\n")
    print(f"\n-------------------------------------")

    # Calculate baseline errors
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(preds.y_actual)
    print(f"\nBaseline RMSE: {RMSE_baseline:.2f}\n")
    print(f"\n-------------------------------------")

    # Check if the model is better than the baseline
    print(f"\nIs the model better than the baseline? {better_than_baseline(preds.y_actual, preds.y_hat)}")

# ============================ model function =============================
def model_xy():
# Wrangle the data
    df = wrangle_zillow()

    # Drop categorical features
    df = df.drop(columns=['property_county_landuse_code', 'property_zoning_desc', 'n-prop_type', 'n-av_room_size', 'state'])

    # Train-test split
    train, val, test = train_val_test(df)

    # Split data into X and y for train and val
    X_train, y_train = xy_split(train, 'home_value')
    X_val, y_val = xy_split(val, 'home_value')
    X_test, y_test = xy_split(test, 'home_value')

    return X_train, y_train, X_val, y_val, X_test, y_test


def model_1(X_train, y_train, X_val, y_val):
    # Initialize the RandomForestRegressor
    rfr = RandomForestRegressor()
    
    # Train the model on the training data
    rfr.fit(X_train, y_train)
    
    # Make predictions on training and validation sets
    train_preds = rfr.predict(X_train)
    val_preds = rfr.predict(X_val)
    
    # Calculate RMSE for training and validation sets
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    # Calculate R-squared (R2) for training and validation sets
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    # Print the metrics
    print(f"\n-------------------------------------")
    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    
    return rfr

# Example usage:
# rfr = RandomForestRegressor()
# trained_model = train_and_evaluate_model(rfr, X_train, y_train, X_val, y_val)

# ======================================= model 2 ============================================\


def model_2(df, target_column, X_val, y_val, early_stopping_rounds=10, params=None):

    # acquire data
    df = wrangle_zillow()

    # Drop categorical features
    df = df.drop(columns=['property_county_landuse_code', 'property_zoning_desc', 'n-prop_type', 'n-av_room_size', 'state'])
    
    # Train-test split
    train, val, test = train_val_test(df)

    # Split data into X and y
    X_train, y_train = xy_split(df, target_column)
    
    # Define the hyperparameters for your XGBoost model (or pass them as an argument)
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 4,
            'early_stopping_rounds': early_stopping_rounds,
            # Add other hyperparameters as needed
        }

    # Define weight data (you can replace this with your actual weights)
    sample_weights = np.ones(X_train.shape[0])  # Example: All weights are set to 1
    
    # Create the XGBoost regressor with your specified hyperparameters
    xgb = XGBRegressor(**params)
    
    # Fit the model to your training data with eval_set and verbose
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, sample_weight=sample_weights)

    # Access the best iteration and best score
    best_iteration = xgb.best_iteration
    best_score = xgb.best_score
    
    # Make predictions on validation set
    val_preds = xgb.predict(X_val)
    
    # Calculate RMSE and R2 for the validation set
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_r2 = r2_score(y_val, val_preds)
    
    # Create a dictionary to store the results
    results = {
        'model': xgb,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'best_iteration': best_iteration,
        'best_score': best_score
    }
    
    # Print the metrics within the function
    print(f"\n-------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Iteration: {best_iteration}")
    print(f"\n-------------------------------------")
    print(f"\nBest Score: {best_score}")
    
    return results


# ========================================= model 3 ================================================

def model_3():

        # acquire data
        df = wrangle_zillow()
    
        # Drop categorical features
        df = df.drop(columns=['parcel_id',
                              'property_county_landuse_code', 
                              'property_zoning_desc', 
                              'n-prop_type', 
                              'n-av_room_size', 
                              'state',
                              'region_id_county',
                              'region_id_zip',
                              'latitude',
                              'longitude',
                              'lot_area',
                              'bedrooms',
                              'basement_sqft',
                              'fips',
                              'rooms',
                              'num_stories',
                              'year_built',
                              'property_landuse_type_id',
                              'ac_type_id',
                              'building_quality_type_id',
                              'heating_or_system_type_id',
                              'deck_type_id',
                              'unit_cnt',
                              'garage_car_cnt',
                              'garage_total_sqft',
                              'pool_cnt',
                              'pool_size_sum',
                              'pool_type_id_2',
                              'pool_type_id_7',
                              'fire_place_cnt',
                              'fire_place_flag',
                              'has_hot_tub_or_spa',
                              'patio_sqft',
                              'storage_sqft',
                              'tax_delinquency_flag',
                              'raw_census_tract_and_block',
                              'n-life',
                              'n-living_area_error',
                              'n-living_area_prop',
                              'n-living_area_prop2',
                              'n-extra_space',
                              'n-extra_space-2',
                              'n-gar_pool_ac',
                              'n-location',
                              'n-location-2',
                              'n-location-2round',
                              'n-latitude-round',
                              'n-longitude-round',
                              'n-zip_count',
                              'n-county_count',                                    
                              'n-ac_ind',
                              'n-heat_ind',
                              'Small',
                              'Medium',
                              'Large'])
        
        # Train-test split
        train, val, test = train_val_test(df)
        
        # Split subsets into X and y only for train and val, not doing test just yet
        X_train, y_train = xy_split(train, 'home_value')
        X_val, y_val = xy_split(val, 'home_value')
        
        # Calculate mean and median of y_train
        y_train_mean = y_train.mean()
        y_train_median = y_train.median()
        
        # Create a DataFrame with y_train statistics
        bl = pd.DataFrame({"y_actual" : y_train,
                           "y_mean" : y_train_mean,
                           "y_median" : y_train_median})
        
        # Apply polynomial feature transformation
        poly = PolynomialFeatures()
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)

        
    
        # Train a Linear Regression model and evaluate it
        lm = LinearRegression()
        trained_model, train_rmse, val_rmse = train_and_evaluate_model(lm, X_train, y_train, X_val, y_val)
        
        # Return the trained model and evaluation metrics
        return {
            'model': trained_model,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'y_train_mean': y_train_mean,
            'y_train_median': y_train_median,
            'y_train_stats': bl
        }

# ======================== final model and visualization ===============================
def final_model(df, target_column, X_test, y_test, early_stopping_rounds=10, params=None):
    
    # acquire data
    df = wrangle_zillow()

    # Drop categorical features
    df = df.drop(columns=['property_county_landuse_code', 'property_zoning_desc', 'n-prop_type', 'n-av_room_size', 'state'])
    
    # Train-test split
    train, val, test = train_val_test(df)

    # Split data into X and y
    X_train, y_train = xy_split(df, target_column)
    X_test, y_test = xy_split(df, target_column)
    
    # Define the hyperparameters for your XGBoost model (or pass them as an argument)
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 4,
            'early_stopping_rounds': early_stopping_rounds,
            # Add other hyperparameters as needed
        }


    # Define weight data (you can replace this with your actual weights)
    sample_weights = np.ones(X_train.shape[0])  # Example: All weights are set to 1
    
    # Create the XGBoost regressor with your specified hyperparameters
    xgb = XGBRegressor(**params)
    
    # Fit the model to your training data with eval_set and verbose
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True, sample_weight=sample_weights)

    # Access the best iteration and best score
    best_iteration = xgb.best_iteration
    best_score = xgb.best_score
    
    # Make predictions on validation set
    test_preds = xgb.predict(X_test)

    # Create a scatter plot of actual vs. predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=test_preds, alpha=0.5, color='orange')
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()
    
    # Calculate RMSE and R2 for the validation set
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)
    
    # Create a dictionary to store the results
    results = {
        'model': xgb,
        'val_rmse': test_rmse,
        'val_r2': test_r2,
        'best_iteration': best_iteration,
        'best_score': best_score
    }
    
    # Print the metrics within the function
    print(f"\n-------------------------------------")
    print(f"\nTest RMSE: {test_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTest R-squared (R2): {test_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Iteration: {best_iteration}")
    print(f"\n-------------------------------------")
    print(f"\nBest Score: {best_score}")
    
    return results
