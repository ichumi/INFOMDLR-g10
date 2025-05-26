# By Izumi Alatriste  K-Fold, Oversampling, Data Augmentation, LeakyReLU
#Stratified K-Fold Cross Validation 
  # Trying with 5 folds now last time the fold 2 was the best performing
# the model doesnt perfom exactly the same depending on what batch is given but 
# its performing well and makes sense as is corresponding to the data.
# Try 3-fold
 

import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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
warnings.filterwarnings("ignore")

# Data Loading and  Preprocessing 

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

# Data Augmentation Functions 

def augment_data(X, y, noise_std=0.01, n_copies=1):
    X_aug, y_aug = [], []
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_std, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y.copy())
    # Concatenate all copies
    X_aug = np.concatenate(X_aug, axis=0)
    y_aug = np.concatenate(y_aug, axis=0)
    return X_aug, y_aug

def time_shift(X, max_shift=10):
    X_aug = np.empty_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift)
        X_aug[i] = np.roll(X[i], shift, axis=1)
    return X_aug

# Random Oversampling Function 

def oversample_minority_classes(X, y):
    unique, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    X_list, y_list = [], []
    for label in unique:
        X_class = X[y == label]
        y_class = y[y == label]
        X_res, y_res = resample(X_class, y_class,
                                replace=True,
                                n_samples=max_count,
                                random_state=42)
        X_list.append(X_res)
        y_list.append(y_res)
    X_balanced = np.concatenate(X_list, axis=0)
    y_balanced = np.concatenate(y_list, axis=0)
    return X_balanced, y_balanced

#  TCN Model with LeakyReLU 

def build_tcn_model(input_shape, num_classes):
    leaky_relu = layers.LeakyReLU(alpha=0.1)
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Permute((2, 1))(inputs)
    for dilation in [1, 2]:
        x = layers.Conv1D(32, 3, dilation_rate=dilation, padding='causal')(x)
        x = layers.BatchNormalization()(x)
        x = leaky_relu(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32)(x)
    x = leaky_relu(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

base_folder = r"./Final Project data/Intra/train"
X, y, _ = load_folder_data(base_folder)
num_classes = 4
input_shape = X.shape[1:]

# Stratified K-Fold Cross Validation 
n_splits = 3
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def plot_training_curves(history, fold):
    metrics = history.history
    epochs = range(1, len(metrics['accuracy']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(epochs, metrics[metric], label='Train')
        ax.plot(epochs, metrics[f'val_{metric}'], label='Val')
        ax.set_title(f'Fold {fold} - Training & Validation {ylabel}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
    plt.tight_layout()
    plt.show()

# Train and Evaluate with Cross-Validation 

fold = 1
results = []
for train_idx, val_idx in kfold.split(X, y):
    print(f"\n----- Fold {fold} ------")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train_os, y_train_os = oversample_minority_classes(X_train, y_train)
    
    # Data augmentation
    X_train_aug, y_train_aug = augment_data(X_train_os, y_train_os, noise_std=0.05, n_copies=4)
    X_train_aug = time_shift(X_train_aug, max_shift=15)

    y_train_cat = to_categorical(y_train_aug, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    model = build_tcn_model(input_shape=input_shape, num_classes=num_classes)

    # Class weights for augmented training set
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_aug), y=y_train_aug)
    class_weights_dict = dict(enumerate(class_weights))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0)

    history = model.fit(
        X_train_aug, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=30,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )

    # Evaluating Model
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    results.append(val_acc)

    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val_cat, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Rest", "Motor", "Story_Math", "Memory"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Rest", "Motor", "Story_Math", "Memory"],
                yticklabels=["Rest", "Motor", "Story_Math", "Memory"])
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    plot_training_curves(history, fold)
    fold += 1

# Mean and std of validation accuracy across folds 
print(f"\nAverage Validation Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")

# Plot
def plot_training_curves(history):
    metrics = history.history
    epochs = range(1, len(metrics['accuracy']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(epochs, metrics[metric], label='Train')
        ax.plot(epochs, metrics[f'val_{metric}'], '', label='Val')
        ax.set_title(f'Trainin & Validation {ylabel}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
    plt.tight_layout()
    plt.show()
