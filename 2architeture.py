import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load data
data = pd.read_csv('output.csv')

# Separate predictors and responses
X = data.drop(['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6'], axis=1).values
y = data[['target1', 'target2', 'target3', 'target4', 'target5']].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

# Fit and transform the target variable
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Define updated ranges for number of hidden layers and units
hidden_layers_range = range(1, 2, 3, 4, 5)
units_range = [350, 370, 400, 420, 450]

best_model = None
best_mse = float('inf')
best_r2 = float('-inf')
best_config = None

# Iterate through different configurations
for num_layers in hidden_layers_range:
    for num_units in units_range:
        # Construct the neural network model
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(num_units, activation='relu'))
        model.add(Dropout(0.2))
        for _ in range(num_layers - 1):
            model.add(Dense(num_units, activation='relu'))
            model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1]))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)

        # Evaluate the model on the validation data
        y_pred = model.predict(X_val)

        # Calculate MSE
        mse = mean_squared_error(y_val, y_pred)

        # Calculate R-squared
        r2 = r2_score(y_val, y_pred)

        # Update best model if current configuration yields better MSE and R-squared
        if mse < best_mse and r2 > best_r2:
            best_mse = mse
            best_r2 = r2
            best_model = model
            best_config = (num_layers, num_units)

# Print the best model's configuration and MSE
print("Best Model Configuration:")
if best_model:
    best_model.summary()
    print(f"Best Mean Squared Error on Validation Set: {best_mse}")
    print(f"Best R-squared: {best_r2}")
    print(f"Best Configuration - Number of Layers: {best_config[0]}, Number of Units per Layer: {best_config[1]}")
else:
    print("No valid model configuration found.")

# Inverse transform predictions and actual values
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_val_inv = scaler_y.inverse_transform(y_val)

# Calculate R-squared for the best model
r2 = r2_score(y_val_inv, y_pred_inv, multioutput='uniform_average')
print(f"R-squared (inversed): {r2}")

# Plot true vs. predicted values
import matplotlib.pyplot as plt

for i in range(y_val_inv.shape[1]):
    plt.figure()
    plt.scatter(y_val_inv[:, i], y_pred_inv[:, i])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs. Predicted Values for target {i+1}')
    plt.show()
