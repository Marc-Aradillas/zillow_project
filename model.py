from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from math import sqrt



# ------------------------ SCALE DATA FUNCTION --------------------
def scale_data(train, val, test, to_scale):
    # make copies for scaling
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
    
    # Print the results
    print(f'The train RMSE is {train_rmse:.2f}.\n')
    print(f'The validation RMSE is {val_rmse:.2f}.\n\n')
    
    return model, train_rmse, val_rmse

# Example usage:
# trained_model, train_rmse, val_rmse = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)