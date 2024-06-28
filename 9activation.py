import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load data
data = pd.read_csv('output.csv')

# Separate predictors and responses
X = data.drop(['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6'], axis=1).values
y = data[['target1', 'target2', 'target3', 'target4', 'target5']].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Define activation functions to test
activation_dict = {
    'relu': 'relu',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
}

best_activation = None
best_mse = float('inf')
best_r2 = float('-inf')
best_model = None

# Iterate through different activation functions
for activation_name, activation_value in activation_dict.items():
    # Define the model
    model = Sequential()
    model.add(Dense(420, input_dim=X_train.shape[1], activation=activation_value, kernel_initializer=RandomNormal()))
    model.add(Dense(420, activation=activation_value, kernel_initializer=RandomNormal()))
    model.add(Dense(420, activation=activation_value, kernel_initializer=RandomNormal()))
    model.add(Dense(y_train.shape[1]))

    # Compile the model
    optimizer = Adam(learning_rate=0.035)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Update the best model if the current one is better
    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_activation = activation_name
        best_model = model

# Print the best activation function and metrics
print(f"Best Activation Function: {best_activation}")
print(f"Best Mean Squared Error on Test Set: {best_mse}")
print(f"Best R-squared: {best_r2}")
