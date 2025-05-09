import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import scipy.io

# Read mat files
mat = scipy.io.loadmat("Xtrain.mat")
xtrain = mat['Xtrain'].squeeze()
test_mat = scipy.io.loadmat("Xtest.mat")
xtest = test_mat['Xtest'].squeeze()

# Scale data to [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(xtrain.reshape(-1, 1)).flatten()
scaled_test = scaler.transform(xtest.reshape(-1, 1)).flatten()  

def create_dataset(series, lookback):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)

# Parameter tuning for lookback values between 10 and 100
lookbacks = list(range(10, 101, 10))  # steps of 10
best_mse = float('inf')
best_lookback = None
best_model = None

lookback_results = []
mse_results = []

for lookback in lookbacks:
    X, y = create_dataset(scaled_data, lookback)

    # CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(lookback, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)  # Predict one step ahead
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32) 

    pred = model.predict(X, verbose = 0).reshape(-1, 1)
    y_true = y.reshape(-1, 1)
    pred_inv = scaler.inverse_transform(pred)
    y_inv = scaler.inverse_transform(y_true)

    # MSE and MAE
    mse = mean_squared_error(y_inv, pred_inv)
    mse_results.append(mse)
    lookback_results.append(lookback)

    # Check if current model performs better
    if mse < best_mse:
        best_mse = mse
        best_lookback = lookback
        best_model = model


print(f"Best lookback: {best_lookback}, MSE: {best_mse:.4f}")

# Make predictions based on best lookback value
input_ord = scaled_data[-best_lookback:].tolist()  # Use last `lookback` values
next_200_preds = []  # list for the predictions
for _ in range(200):  # Loop to predict the next 200 data points
    a_input = np.array(input_ord[-best_lookback:]).reshape(1, best_lookback, 1)  # input for prediction
    pred = best_model.predict(a_input, verbose= 0)[0, 0]
    next_200_preds.append(pred)
    input_ord.append(pred)

pred_inverse = scaler.inverse_transform(np.array(next_200_preds).reshape(-1, 1))

real_values = scaled_test[:200]
real_values_inverse = scaler.inverse_transform(real_values.reshape(-1, 1))

test_mse = mean_squared_error(real_values_inverse, pred_inverse)
test_mae = mean_absolute_error(real_values_inverse, pred_inverse)

print(f"\nTest MSE (comparison of 200 predicted points with real test values): {test_mse:.4f}")
print(f"Test MAE (comparison of 200 predicted points with real test values): {test_mae:.4f}")

#######################

# Prediction vs first
plt.figure(figsize=(10, 4))
plt.plot(y_inv, label='Actual')
plt.plot(pred_inv, label='Predicted', alpha=0.7)
plt.title(f'1-step Ahead Prediction (Lookback={lookback})')
plt.xlabel('Time Step')
plt.ylabel('Laser Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

# MSE and Lookback plot
plt.figure(figsize=(8, 4))
plt.plot(lookback_results, mse_results, marker='o')
plt.title('MSE vs. Lookback Window Size')
plt.xlabel('Lookback (Time Steps)')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.tight_layout()
plt.show()

# Prediction vs actual values plot (next 200 data points)
plt.figure(figsize=(10, 4))
plt.plot(pred_inverse, label='Predictions')
plt.plot(real_values_inverse, label="Actual")
plt.title(f'Prediction of next 200 data points (Best Lookback={best_lookback})')
plt.xlabel('Step Time')
plt.ylabel('Predicted Laser Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
