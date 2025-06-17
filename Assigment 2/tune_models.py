import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna

X = np.load('X_meg.npy')
y = np.load('y_meg.npy')

X = np.array([StandardScaler().fit_transform(x.T).T for x in X])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
num_classes = len(np.unique(y))
input_shape = X_train.shape[1:]

y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val,   num_classes)

def build_tcn_model(input_shape, num_classes, filters, kernel_size, dense_units):
    inp = layers.Input(shape=input_shape)
    x = layers.Permute((2,1))(inp)
    for d in [1,2]:
        x = layers.Conv1D(filters, kernel_size, dilation_rate=d, padding='causal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out)


def build_lstm_model(input_shape, num_classes, lstm_units, dense_units):
    inp = layers.Input(shape=input_shape)
    x = layers.Permute((2,1))(inp)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dense(dense_units)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out)

# optuna
def objective_tcn(trial):
    filters     = trial.suggest_categorical('filters', [16, 32, 64])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    dense_units = trial.suggest_categorical('dense_units', [32, 64])
    lr          = trial.suggest_loguniform('lr_tcn', 1e-4, 1e-2)
    model = build_tcn_model(input_shape, num_classes, filters, kernel_size, dense_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=5, batch_size=32, verbose=0)
    return history.history['val_accuracy'][-1]


def objective_lstm(trial):
    lstm_units  = trial.suggest_categorical('lstm_units', [32, 64, 128])
    dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
    lr          = trial.suggest_loguniform('lr_lstm', 1e-4, 1e-2)
    model = build_lstm_model(input_shape, num_classes, lstm_units, dense_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=5, batch_size=32, verbose=0)
    return history.history['val_accuracy'][-1]

study_tcn = optuna.create_study(direction='maximize')
study_tcn.optimize(objective_tcn, n_trials=12)
print('TCN best:', study_tcn.best_params, 'acc=', study_tcn.best_value)

study_lstm = optuna.create_study(direction='maximize')
study_lstm.optimize(objective_lstm, n_trials=12)
print('LSTM best:', study_lstm.best_params, 'acc=', study_lstm.best_value)

