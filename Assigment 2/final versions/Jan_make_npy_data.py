#!/usr/bin/env python3
import h5py
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


# point to exactly the folder you want to preprocess:
base_folder  = "Final Project data"
train_folder = os.path.join(base_folder, "cross", "train")  # or "Intra"/"train"

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


# load & preprocess:
X, y, info = load_folder_data(train_folder)
print(f"Loaded {X.shape[0]} samples with shape {X.shape[1:]}")

# save out .npy files:
np.save("X_meg.npy", X)
np.save("y_meg.npy", y)