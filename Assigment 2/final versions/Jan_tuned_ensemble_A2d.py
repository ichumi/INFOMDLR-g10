# By Jan Stelmaszczyk based on code from Izumi Alatriste
# Added a lstm model and combined results from both tcn and lstm using soft-voting (ensemble function)
# Added optuna search for lstm hyperparameters
# Code contains some other methods for ensembling models: 
# (grid search for optimal model weights or classwise weights)
# that have been commented out.

import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import optuna
from sklearn.metrics import f1_score
import itertools

warnings.filterwarnings("ignore")

# tuned hyperparameters
best_tcn   = {'filters':32,'kernel_size':5,'dense_units':32,'lr':0.00176}
best_lstm  = {'lstm_units':128,'dense_units':32,'lr':0.00045}


data_mode = "cross"  # "intra" or "cross" MANUALLY CHANGE!!!!

base_folder = "Final Project data"  # Exactly as shown in the folder

if data_mode == "intra":
    train_folder = os.path.join(base_folder, "Intra", "train")
    test_folder = os.path.join(base_folder, "Intra", "test")
else:
    train_folder = os.path.join(base_folder, "Cross", "train")
    test_folders = [
        os.path.join(base_folder, "Cross", "test1"),
        os.path.join(base_folder, "Cross", "test2"),
        os.path.join(base_folder, "Cross", "test3"),
    ]


# Helper functions 
def get_dataset_name(file_path):
    return '_'.join(os.path.basename(file_path).replace('.h5', '').split('_')[:-1])

def load_and_normalize_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        dataset_name = get_dataset_name(file_path)
        matrix = f[dataset_name][()]
        return StandardScaler().fit_transform(matrix.T).T

def get_label_from_filename(fname):
    if fname.startswith("rest"): return 0
    if fname.startswith("task_motor"): return 1
    if fname.startswith("task_story_math"): return 2
    if fname.startswith("task_working_memory"): return 3
    return -1

def downsample(matrix, target_length=512):
    return matrix[:, ::(matrix.shape[1] // target_length)][:, :target_length]

def load_folder_data(folder_path):
    X, y, info = [], [], []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    for fname in os.listdir(folder_path):
        if fname.endswith('.h5'):
            path = os.path.join(folder_path, fname)
            try:
                mat = load_and_normalize_h5(path)
                X.append(downsample(mat))
                y.append(get_label_from_filename(fname))
                info.append((fname, mat.shape))
            except Exception as e:
                info.append((fname, f"Error: {e}"))
    return np.stack(X), np.array(y), info

def augment_data(X, y, noise_std=0.01, n_copies=1):
    X_aug, y_aug = [], []
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_std, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y.copy())
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

def time_shift(X, max_shift=10):
    X_aug = np.empty_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift)
        X_aug[i] = np.roll(X[i], shift, axis=1)
    return X_aug

def oversample_minority_classes(X, y):
    unique, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    X_list, y_list = [], []
    for label in unique:
        X_class = X[y == label]
        y_class = y[y == label]
        X_res, y_res = resample(X_class, y_class, replace=True, n_samples=max_count, random_state=42)
        X_list.append(X_res)
        y_list.append(y_res)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

X = np.load('X_meg.npy')
y = np.load('y_meg.npy')
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)
num_classes = 4
input_shape = X_train.shape[1:]

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

# Model Builders
def build_tcn_model(input_shape, num_classes, filters=64, kernel_size=5, dense_units=32):
    leaky_relu = layers.LeakyReLU(alpha=0.1)
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Permute((2, 1))(inputs)
    for dilation in [1, 2]:
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                          dilation_rate=dilation, padding='causal')(x)
        x = layers.BatchNormalization()(x)
        x = leaky_relu(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units)(x)
    x = leaky_relu(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_lstm_model(input_shape, num_classes, lstm_units=64, dense_units=32, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=input_shape)
    
    # permute to (time_steps, features)
    x = layers.Permute((2, 1))(inputs)
    
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(dense_units)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

'''
# Optuna tuning for TCN
def opt_tune_tcn(trial):
    filters = trial.suggest_categorical('filters', [16, 32, 64])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    dense_units= trial.suggest_categorical('dense_units', [32, 64])
    
    model = build_tcn_model(input_shape=input_shape,
                            num_classes=num_classes,
                            filters=filters,
                            kernel_size=kernel_size,
                            dense_units= dense_units
                            )
    
    tuning = model.fit(X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=5,
                        batch_size=32,
                        verbose=0, shuffle=False)
    val_acc = tuning.history['val_accuracy'][-1]
    print(f"Trial {trial.number}: filters={filters}, kernel_size={kernel_size},dense_units={dense_units} val_accuracy={val_acc:.4f}")
    return val_acc

optuna_tune = optuna.create_study(direction='maximize')
optuna_tune.optimize(opt_tune_tcn, n_trials=15)

print("Results")
print("The best performing TCN hyperparameters:", optuna_tune.best_params)
print(f"According to the highest validation accuracy receives from the trials: {optuna_tune.best_value:.4f}")
'''

'''
# Optuna tuning for LSTM
def opt_tune_lstm(trial):
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64])
    dense_units= trial.suggest_categorical('dense_units', [32, 64])
    
    model = build_lstm_model(input_shape=input_shape,
                            num_classes=num_classes,
                            lstm_units=lstm_units,
                            dense_units= dense_units
                            )
    
    tuning = model.fit(X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=5,
                        batch_size=32,
                        verbose=0, shuffle=False)
    val_acc = tuning.history['val_accuracy'][-1]
    print(f"Trial {trial.number}: lstm_units={lstm_units},dense_units={dense_units} val_accuracy={val_acc:.4f}")
    return val_acc

optuna_tune = optuna.create_study(direction='maximize')
optuna_tune.optimize(opt_tune_lstm, n_trials=15)

print("Results")
print("The best performing LSTM hyperparameters:", optuna_tune.best_params)
print(f"According to the highest validation accuracy receives from the trials: {optuna_tune.best_value:.4f}")
'''

'''
# Definition of a search function for best weights per ensemble
def find_best_ensemble_weights(model_tcn, model_lstm, X_val, y_val, metric='accuracy'):
    best_score = -1
    best_weights = (0.5, 0.5)
    y_true = y_val

    probs_tcn = model_tcn.predict(X_val)
    probs_lstm = model_lstm.predict(X_val)

    for w_tcn in np.linspace(0, 1, 11):
        w_lstm = 1 - w_tcn
        probs_ens = w_tcn * probs_tcn + w_lstm * probs_lstm
        preds = np.argmax(probs_ens, axis=1)

        if metric == 'accuracy':
            score = np.mean(preds == y_true)
        elif metric == 'macro_f1':
            score = f1_score(y_true, preds, average='macro')
        else:
            raise ValueError("Invalid metric: choose 'accuracy' or 'macro_f1'")

        if score > best_score:
            best_score = score
            best_weights = (w_tcn, w_lstm)

    print(f"Best weights: TCN={best_weights[0]:.2f}, LSTM={best_weights[1]:.2f} ({metric} = {best_score:.3f})")
    return best_weights
'''

'''
# Definition of a search for best weights per class
def find_best_classwise_weights(y_true, probs_tcn, probs_lstm):
    num_classes = probs_tcn.shape[1]
    best_score = 0
    best_weights = None
    
    # Try weights from 0 to 1 in 0.2 steps for each class
    for tcn_w in itertools.product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], repeat=num_classes):
        tcn_w = np.array(tcn_w)
        lstm_w = 1.0 - tcn_w  # inverse per class
        combined = probs_tcn * tcn_w + probs_lstm * lstm_w
        preds = np.argmax(combined, axis=1)
        score = f1_score(y_true, preds, average='macro')
        
        if score > best_score:
            best_score = score
            best_weights = tcn_w
    
    return best_weights, 1.0 - best_weights, best_score
'''

def plot_training_curves(history):
    metrics = history.history
    epochs = range(1, len(metrics['accuracy']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(epochs, metrics[metric], label='Train')
        ax.plot(epochs, metrics[f'val_{metric}'], label='Val')
        ax.set_title(f'Training & Validation {ylabel}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
    plt.tight_layout()
    plt.show()

# Training 
print("=" * 60)
print(f"MODE: {'INTRA-SUBJECT' if data_mode == 'intra' else 'CROSS-SUBJECT'}") 
print("=" * 60)

X_train_raw, y_train_raw, _ = load_folder_data(train_folder)
print(f"[TRAIN] Samples: {X_train_raw.shape[0]}")
print(f"[TRAIN] Class distribution: {dict(zip(*np.unique(y_train_raw, return_counts=True)))}")

# Oversample & augment training data
X_train_os, y_train_os = oversample_minority_classes(X_train_raw, y_train_raw)
X_train_aug, y_train_aug = augment_data(X_train_os, y_train_os, noise_std=0.05, n_copies=4)
X_train_aug = time_shift(X_train_aug, max_shift=15)

y_train_cat = to_categorical(y_train_aug, num_classes=4)
input_shape = X_train_aug.shape[1:]

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weights_dict = dict(enumerate(class_weights))

# Build & compile both models
print("Building TCN and LSTM models...")
model_tcn = build_tcn_model(input_shape=input_shape, num_classes=num_classes)
model_lstm = build_lstm_model(input_shape=input_shape, num_classes=num_classes)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
]

# Train TCN
print("\nTraining TCN model...")
history_tcn = model_tcn.fit(
    X_train_aug, to_categorical(y_train_aug, num_classes),
    validation_split=0.2,
    epochs=30,
    batch_size=8,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# Train LSTM
print("\nTraining LSTM model...")
history_lstm = model_lstm.fit(
    X_train_aug, to_categorical(y_train_aug, num_classes),
    validation_split=0.2,
    epochs=30,
    batch_size=8,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)


# Simple averaging ensemble 
def evaluate_ensemble(X_test, y_test, models, weights=None):
    
    probs = None
    for model, w in models:
        p = model.predict(X_test)
        if probs is None:
            probs = w * p
        else:
            probs += w * p
    preds = np.argmax(probs, axis=1)
    print(classification_report(y_test, preds, target_names=["Rest", "Motor", "Story_Math", "Memory"]))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Rest","Motor","Story_Math","Memory"],
                yticklabels=["Rest","Motor","Story_Math","Memory"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Ensemble Confusion Matrix"); plt.show()
    plot_training_curves(history_tcn)
    plot_training_curves(history_lstm)

'''
# Implementation of a search for best weights per ensemble
probs_tcn = model_tcn.predict(X_val)
probs_lstm = model_lstm.predict(X_val)

best_tcn_w, best_lstm_w, best_score = find_best_classwise_weights(y_val, probs_tcn, probs_lstm)
print("TCN weights per class:", best_tcn_w)
print("LSTM weights per class:", best_lstm_w)

best_w = find_best_ensemble_weights(model_tcn, model_lstm, X_val, y_val, metric='macro_f1')

models_to_ensemble = [
    (model_tcn, best_w[0]),
    (model_lstm, best_w[1])
]
'''
    
# Model weights for soft-voting
models_to_ensemble = [
    (model_tcn, 0.65),
    (model_lstm, 0.35)
]

'''
# Implementation of a search for best weights per class
def classwise_ensemble(tcn_probs, lstm_probs, tcn_w, lstm_w):
    return np.argmax(tcn_probs * tcn_w + lstm_probs * lstm_w, axis=1)

y_pred = classwise_ensemble(probs_tcn, probs_lstm, best_tcn_w, best_lstm_w)

def evaluate_classwise_ensemble(X_test, y_test, model_tcn, model_lstm, tcn_w, lstm_w):
    tcn_probs = model_tcn.predict(X_test)
    lstm_probs = model_lstm.predict(X_test)
    
    y_pred = classwise_ensemble(tcn_probs, lstm_probs, tcn_w, lstm_w)
    
    print(classification_report(y_test, y_pred, target_names=["Rest", "Motor", "Story_Math", "Memory"]))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Rest", "Motor", "Story_Math", "Memory"],
                yticklabels=["Rest", "Motor", "Story_Math", "Memory"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Classwise Ensemble Confusion Matrix")
    plt.show()
'''

if data_mode == "intra":
    X_test, y_test, _ = load_folder_data(test_folder)
    evaluate_ensemble(X_test, y_test, models_to_ensemble)
#    evaluate_classwise_ensemble(X_test, y_test, model_tcn, model_lstm, best_tcn_w, best_lstm_w)
    
else:
    for test_path in test_folders:
        X_test, y_test, _ = load_folder_data(test_path)
        print(f"\n=== Ensemble on {os.path.basename(test_path)} ===")
        evaluate_ensemble(X_test, y_test, models_to_ensemble)
#        evaluate_classwise_ensemble(X_test, y_test, model_tcn, model_lstm, best_tcn_w, best_lstm_w)
