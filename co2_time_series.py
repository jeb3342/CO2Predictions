




import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset from a CSV file
file_path = '/Users/jeb_bryson12/Documents/School Documents/Spring 2024/STOR 556/Time Series Python/co2 copy.csv'  # Update this to the path of your CSV file
df = pd.read_csv(file_path)

# Combine 'year' and 'month' into a single 'date' column
df['date'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])

# Feature Engineering: Creating lag features
df.sort_values('date', inplace=True)
for i in range(1, 13):
    df[f'lag_{i}'] = df['average'].shift(i)

# Remove rows with NaN values (the first 12 months now have missing lagged values)
df.dropna(inplace=True)

# Define features and target
X = df[[f'lag_{i}' for i in range(1, 13)]]
y = df['average']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

# Example: Predict the next month's CO2 level (you need to replace this part with your actual data)
# Assuming 'df' has the latest 12 months of data at the bottom
next_month_features = df[[f'lag_{i}' for i in range(1, 13)]].iloc[-1].shift(-1)
next_month_features['lag_1'] = df['average'].iloc[-1]  # Last known value as the most recent lag
next_month_prediction = model.predict(next_month_features.values.reshape(1, -1))

print(f"Predicted CO2 level for next month: {next_month_prediction[0]}")












#Refining the model with GridsearchCV
    #Importing the relevant Packages
from sklearn.model_selection import GridSearchCV


# Assuming df is your DataFrame after preprocessing
X = df[[f'lag_{i}' for i in range(1, 13)]]  # Features
y = df['average']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score (RMSE): {mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test))**0.5}")










#Predicting Next Months Value with XGboost given new tuned gridsearch parameters

# Assuming grid_search has been executed and has the best parameters
best_params = grid_search.best_params_

# Initialize a new XGBoost model with the best parameters
final_model = xgb.XGBRegressor(**best_params)

# Train the model on the entire dataset (or just the training set)
final_model.fit(X, y)  # X and y should be defined as your feature matrix and target vector

# Assuming df is your DataFrame and you have prepared your features in the same way as before
# Prepare the features for the next month's prediction
# This step requires creating a feature vector for the next month, similar to your existing feature preparation
# For simplicity, let's assume 'next_month_features' is prepared and contains the correct feature values
next_month_features = df[[f'lag_{i}' for i in range(2, 13)] + ['average']].iloc[-1].values.reshape(1, -1)
# Note: Adjust the feature preparation as per your dataset and the lag features you've used

# Make the prediction for the next month
next_month_prediction = final_model.predict(next_month_features)

print(f"Predicted CO2 level for the next month: {next_month_prediction[0]}")