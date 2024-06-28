import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, GlorotUniform
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Seed Initialization
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# Load data
data = pd.read_csv('output.csv')

# Separate predictors and responses
X = data.drop(['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6'], axis=1).values
y = data[['target1', 'target2', 'target3', 'target4', 'target5']].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Define dropout rates to test
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

best_dropout_rate = None
best_mse = float('inf')
best_r2 = float('-inf')
best_model = None

# Iterate through different dropout rates
for dropout_rate in dropout_rates:
    # Clear previous models
    tf.keras.backend.clear_session()

    # Define the model
    model = Sequential()

    # Input layer and first hidden layer with HeNormal initialization and relu activation
    model.add(Dense(420, input_dim=X_train.shape[1], activation='relu', kernel_initializer=HeNormal(seed=seed_value)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Second hidden layer
    model.add(Dense(420, activation='relu', kernel_initializer=GlorotUniform(seed=seed_value)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Third hidden layer
    model.add(Dense(420, activation='relu', kernel_initializer=GlorotUniform(seed=seed_value)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(y_train.shape[1]))

    # Compile the model with the current learning rate
    optimizer = Adam(learning_rate=0.035)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    # Calculate MSE and R-squared
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f'Dropout rate: {dropout_rate}, MSE: {mse}, R-squared: {r2}')

    # Update the best model if the current one is better
    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_dropout_rate = dropout_rate
        best_model = model

# Print the best model's configuration and metrics
print(f"Best Dropout Rate: {best_dropout_rate}")
print(f"Best Mean Squared Error on Validation Set: {best_mse}")
print(f"Best R-squared: {best_r2}")

# Plot true vs. predicted values for the best model
y_pred_inv = scaler_y.inverse_transform(best_model.predict(X_test))
plt.figure()
for i in range(y_test_inv.shape[1]):
    plt.scatter(y_test_inv[:, i], y_pred_inv[:, i])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs. Predicted Values for target {i+1}')
    plt.show()
