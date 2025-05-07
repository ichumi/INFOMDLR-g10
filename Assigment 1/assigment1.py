import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import scipy.io

#Read mat
mat = scipy.io.loadmat("Xtrain.mat")
xtrain = mat['Xtrain'].squeeze()

# Scale data to [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(xtrain.reshape(-1, 1)).flatten()
scaled = scaler.fit_transform(xtrain.reshape(-1, 1)).flatten()

#data pairs
def create_dataset(series, lookback):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


#Define lookback(?(we have to cover more))
lookback = 20
X, y = create_dataset(scaled_data, lookback)
X = X.reshape((X.shape[0], X.shape[1], 1))


# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(lookback, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)  # predict one step ahead ()
])


model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32, verbose=0)

#Reescaling 
predic = model.predict(X)
predic_rescaled = scaler.inverse_transform(predic) 
y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

#MSE
mse = mean_squared_error(y_rescaled, predic_rescaled)
mae = mean_absolute_error(y_rescaled, predic_rescaled)
print(f"Lookback: 20 — MSE: {mse:.4f}, MAE: {mae:.4f}")

#Comparing model behaviour 
lookbacks = [5, 10, 20, 30, 50]
results = {}
first_lb = lookbacks[0]
first_pred, first_y_inv = None, None

for lb in lookbacks:
    X, y = create_dataset(scaled, lb)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = tf.keras.Sequential([
        # temporal patterns
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(lb, 1)),
        # Reduce sequence length
        tf.keras.layers.MaxPooling1D(pool_size=2),
        # Flatten vector Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)


    pred = model.predict(X)
    pred_inv = scaler.inverse_transform(pred) 
    y_inv = scaler.inverse_transform(y.reshape(-1, 1))

    #MSE
    mse = mean_squared_error(y_inv, pred_inv)
    mae = mean_absolute_error(y_inv, pred_inv)
    results[lb] = {'MSE': mse, 'MAE': mae}
    print(f"Lookback: {lb} — MSE: {mse:.4f}, MAE: {mae:.4f}")

    if lb == first_lb:
        first_pred = pred_inv
        first_y_inv = y_inv

    #Prediction
    input_ord = scaled[-lb:].tolist() # lookback values in a list
    next_200_preds = [] #list for the predictions
    for _ in range(200): #loop to predict the next 200 data points
        a_input = np.array(input_ord[-lb:]).reshape(1, lb, 1) #the used input to make predictions
        pred = model.predict(a_input, verbose=0)[0, 0]
        next_200_preds.append(pred)
        input_ord.append(pred)

    pred_inverse = scaler.inverse_transform(np.array(next_200_preds).reshape(-1, 1)) # transform to original values
    print(f"\nRecursively predicted 200 data points of lookback {lb}:")
    print(pred_inverse.flatten())     

# Prediction vs first
plt.figure(figsize=(10, 4))
plt.plot(first_y_inv, label='Actual')
plt.plot(first_pred, label='Predicted', linestyle='-')
plt.title(f'CNN Prediction vs Actual (Lookback = )')
plt.xlabel('Step Time')
plt.ylabel('Laser Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#MSE and Lookback plots
plt.plot(results.keys(), [v['MSE'] for v in results.values()], marker='o')
plt.title('MSE vs Lookback Window Size')
plt.xlabel('Lookback (Time Steps)')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.tight_layout()
plt.show()
