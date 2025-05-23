#By Izumi Alatriste

import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#TCN MODEL
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



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


def load_folder_data(folder_path):
    normalized_data_list = []
    labels = []
    file_info = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.h5'):
            file_path = os.path.join(folder_path, fname)
            try:
                norm_matrix = load_and_normalize_h5(file_path)
                norm_ds = downsample(norm_matrix, target_length=512)
                normalized_data_list.append(norm_ds)
                labels.append(get_label_from_filename(fname))
                file_info.append((fname, norm_ds.shape))
            except Exception as e:
                file_info.append((fname, f"Error: {e}"))
    X_data = np.stack(normalized_data_list)
    y_data = np.array(labels)
    return X_data, y_data, file_info

# Read intra subject data
base_folder = r"./Final Project data/Intra/train"
X, y, file_info = load_folder_data(base_folder)
X_expanded = X
y_cat = to_categorical(y, num_classes=4)


intra_test_folder = r"./Final Project data/Intra/test"
X_intra_test, y_intra_test_raw, _ = load_folder_data(intra_test_folder)
y_intra_test = to_categorical(y_intra_test_raw, num_classes=4)


# Check one file for keys
test_file = os.path.join(base_folder, "rest_105923_1.h5")
with h5py.File(test_file, 'r') as f:
    print("File keys:", list(f.keys()))


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

model_intra = build_tcn_model(input_shape=X_train.shape[1:], num_classes=4)
model_intra.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training output
print("\nTraining TCN model on intra-subject data...")
history = model_intra.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=8)

# Evaluation of the intra model 
val_loss, val_acc = model_intra.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Loss Intra: {val_loss:.4f}")
print(f"Validation Accuracy Intra: {val_acc:.4f}")


# Load and preprocess cross-subject training data 
cross_train_folder = r"./Final Project data/Cross/train"
X_cross_train, y_cross_train_raw, cross_file_info = load_folder_data(cross_train_folder)
y_cross_train = to_categorical(y_cross_train_raw, num_classes=4)

cross_test_folders = [
    r"./Final Project data/Cross/test1",
    r"./Final Project data/Cross/test2",
    r"./Final Project data/Cross/test3"
]
X_cross_test = []
y_cross_test = []
for folder in cross_test_folders:
    X_tmp, y_tmp_raw, _ = load_folder_data(folder)
    X_cross_test.append(X_tmp)
    y_cross_test.append(to_categorical(y_tmp_raw, num_classes=4))

X_cross_test = np.concatenate(X_cross_test, axis=0)
y_cross_test = np.concatenate(y_cross_test, axis=0)


cross_shape_info = {
    "X shape (samples, sensors, time)": X_cross_train.shape,
    "y shape (samples,)": y_cross_train_raw.shape,
    "Label distribution": dict(zip(*np.unique(y_cross_train_raw, return_counts=True)))
}

summary_df_cross = pd.DataFrame([cross_shape_info])
print("\nDataset Summary")
print(summary_df_cross)

df_cross = pd.DataFrame(cross_file_info, columns=["Filename", "Shape or Error"])
print("\nFile Summary")
print(df)
print(f"\nX: {X_cross_train.shape}")
print(f"y: {y_cross_train_raw.shape}")


# Split cross-subject training data into train and validation sets
X_cross_train_split, X_cross_val, y_cross_train_split, y_cross_val = train_test_split(
    X_cross_train, y_cross_train, test_size=0.2, random_state=42, stratify=y_cross_train_raw)


# TCN for cross-subject
model_cross = build_tcn_model(input_shape=X_cross_train.shape[1:], num_classes=4)
model_cross.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining TCN model on cross-subject data...")
history_cross = model_cross.fit(
    X_cross_train_split, y_cross_train_split,
    validation_data=(X_cross_val, y_cross_val),
    epochs=15,
    batch_size=8
)

# Evaluation of the cross model 
val_loss_cross, val_acc_cross = model_intra.evaluate(X_cross_val, y_cross_val, verbose=0)
print(f"\nValidation Loss Cross: {val_loss_cross:.4f}")
print(f"Validation Accuracy Cross: {val_acc_cross:.4f}")

#Evaluation on test data 
test_loss_intra, test_acc_intra = model_intra.evaluate(X_intra_test, y_intra_test, verbose=0)
print(f"\nIntra-subject Test Loss: {test_loss_intra:.4f}")
print(f"Intra-subject Test Accuracy: {test_acc_intra:.4f}")

test_loss, test_acc = model_cross.evaluate(X_cross_test, y_cross_test, verbose=0)
print(f"\nCross-subject Test Loss: {test_loss:.4f}")
print(f"Cross-subject Test Accuracy: {test_acc:.4f}")

#PLOT
#for visualization of how the intra model is doing as per evaluation values.
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
