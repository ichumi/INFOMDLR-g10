#By Izumi Alatriste

import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#DATA PREPROCESSING AND DOWNSAMPLING

# dataset name
def get_dataset_name(file_name_with_dir):
    filename = os.path.basename(file_name_with_dir).replace('.h5', '')
    parts = filename.split('_')
    dataset_name = '_'.join(parts[:-1])  
    return dataset_name

# Function to load and Z-normalize
def load_and_normalize_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        dataset_name = get_dataset_name(file_path)
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}. Found keys: {list(f.keys())}")
        matrix = f[dataset_name][()]
        scaler = StandardScaler()
        normalized = scaler.fit_transform(matrix.T).T  # Normalize per sensor
        return normalized

# Label mapping
def get_label_from_filename(filename):
    if filename.startswith("rest"):
        return 0      
    elif filename.startswith("task_motor"):  
        return 1   
    elif filename.startswith("task_story_math"):
        return 2
    elif filename.startswith("task_working_memory"):
        return 3
    return -1

# Downsampling
def downsample(matrix, target_length=512):  
    factor = matrix.shape[1] // target_length
    return matrix[:, ::factor][:, :target_length]

base_folder = r"C:/Users/Yukin/anaconda3/envs/evol/A_2/dl/Final_Project_data/Final Project data/Intra/train"
test_file = os.path.join(base_folder, "rest_105923_1.h5")

with h5py.File(test_file, 'r') as f:
    print("File keys:", list(f.keys()))

normalized_data_list = []
file_info = []
labels = []  

file_names = [f for f in os.listdir(base_folder) if f.endswith('.h5')]

for fname in file_names:
    file_path = os.path.join(base_folder, fname)
    try:
        norm_matrix = load_and_normalize_h5(file_path)
        norm_ds = downsample(norm_matrix, target_length=512)  
        normalized_data_list.append(norm_ds)
        file_info.append((fname, norm_ds.shape))  
        labels.append(get_label_from_filename(fname))
    except Exception as e:
        file_info.append((fname, f"Error: {e}"))

X = np.stack(normalized_data_list)
y = np.array(labels)

assert set(np.unique(y)).issubset({0, 1, 2, 3}), "Unexpected label values"

shape_info = {
    "X shape (samples, sensors, time)": X.shape,
    "y shape (samples,)": y.shape,
    "Label distribution": dict(zip(*np.unique(y, return_counts=True)))
}

summary_df = pd.DataFrame([shape_info])
print("\nDataset Summary")
print(summary_df)

df = pd.DataFrame(file_info, columns=["Filename", "Shape or Error"])
print("\nFile Summary")
print(df)
print(f"\nX: {X.shape}")
print(f"y: {y.shape}")

np.save("X_meg.npy", X)
np.save("y_meg.npy", y)
print("\nSaved X_meg.npy and y_meg.npy to disk.")

#TCN MODEL
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


X_expanded = X
y_cat = to_categorical(y, num_classes=4)

# Train/test 
X_train, X_val, y_train, y_val = train_test_split(X_expanded, y_cat, test_size=0.2, random_state=42, stratify=y)

# TCN block with Conv1D (Why?: We use Conv1D inside a TCN block 
# because it’s the most efficient and close to a biologically plausible way to
# extract temporal patterns in MEG data. 
# It handles both local and long-range time dependencies using dilation, 
# avoids recurrence, and is scalable to longer 
# sequences like our 512-step MEG windows.
#)
def build_tcn_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Permute((2, 1))(inputs) 

    x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=1, padding='causal', activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='causal', activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=4, padding='causal', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
model = build_tcn_model(input_shape=X_train.shape[1:], num_classes=4)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training output
print("\nTraining TCN model...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=8)

# Evaluation of the model 
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

#PLOT
#for visualization of how the model is doing as per evaluation values.
import matplotlib.pyplot as plt
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

plot_training_curves(history)
