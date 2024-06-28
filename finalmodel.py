import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Define the model
model = Sequential()

# Input layer and first hidden layer with RandomNormal initialization and relu activation
model.add(Dense(420, input_dim=X_train.shape[1], activation='relu', kernel_initializer=RandomNormal()))

# Batch normalization
model.add(BatchNormalization())

# Dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Second hidden layer
model.add(Dense(420, activation='relu', kernel_initializer=RandomNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Third hidden layer
model.add(Dense(420, activation='relu', kernel_initializer=RandomNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output layer
model.add(Dense(y_train.shape[1]))

# Compile the model with a different optimizer and learning rate
optimizer = Adam(learning_rate=0.035)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Define a custom accuracy metric (predictions within 10% of actual value)
accuracy_threshold = 0.1
accuracy = np.mean(np.abs((y_pred - y_test) / y_test) < accuracy_threshold)
print(f'Custom accuracy: {accuracy * 100:.2f}%')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
