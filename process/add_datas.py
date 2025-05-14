import pandas as pd
import numpy as np
import random


def translate_data(data, translation_range=0.02):
    data = data.reshape(1, -1)
    translation = np.random.uniform(-translation_range, translation_range, size=(1, data.shape[1]))
    new_data = data + translation
    if new_data.ndim == 3:
        new_data = new_data.reshape(new_data.shape[0], -1)
    return new_data


def add_random_noise(data, noise_scale=0.01):
    noise = np.random.normal(0, noise_scale, size=data.shape)
    new_data = data + noise
    if new_data.ndim == 3:
        new_data = new_data.reshape(new_data.shape[0], -1)
    return new_data


def rotate_data(data, angle_range=25):
    data = data.reshape(1, -1)
    num_points = data.shape[1] // 3
    data_3d = data.reshape(-1, num_points, 3)
    angle = np.deg2rad(np.random.uniform(-angle_range, angle_range))
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    # rotation_matrix = np.array([
    #     [cos_theta, -sin_theta, 0],
    #     [sin_theta, cos_theta, 0],
    #     [0, 0, 1]
    # ])  # z轴
    # rotation_matrix = np.array([
    #     [1, 0, 0],
    #     [0, cos_theta, -sin_theta],
    #     [0, sin_theta, cos_theta]
    # ])  # x轴
    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])  # y轴
    new_data_3d = np.zeros_like(data_3d)
    for sample_idx in range(data_3d.shape[0]):
        for point_idx in range(data_3d.shape[1]):
            point = data_3d[sample_idx, point_idx]
            new_point = np.dot(point, rotation_matrix.T)
            new_data_3d[sample_idx, point_idx] = new_point

    new_data = new_data_3d.reshape(data_3d.shape[0], -1)
    if new_data.shape[0] == 1:
        new_data = new_data.flatten()
    return new_data


def save_augmented_data(target_label, add_num, file_path):
    data = pd.read_csv(file_path)
    label_data = data[data['label'] == target_label]
    labels = label_data['label'].values
    feature_data = label_data.drop(columns=['label'])
    original_samples = feature_data.values
    num_original_samples = len(original_samples)
    augmented_features = []
    for _ in range(add_num):
        sample_index = np.random.randint(0, num_original_samples)
        sample = original_samples[sample_index].copy()
        sample = add_random_noise(sample)
        sample = rotate_data(sample)
        sample = sample.flatten()
        augmented_features.append(sample)
    augmented_df = pd.DataFrame(augmented_features, columns=feature_data.columns)
    selected_labels = np.random.choice(labels, add_num)
    augmented_df['label'] = selected_labels
    with open(file_path, 'a', newline='') as f:
        augmented_df.to_csv(f, header=f.tell() == 0, index=False)


def save_augmented_frames(frames, add_num, file_path):
    augmented_frames = []
    for _ in range(add_num):
        np.random.seed(random.randint(0, 999999))
        for frame in frames:
            features = np.array(frame[:-1], dtype=np.float32).reshape(1, -1)
            label = frame[-1]
            augmented = add_random_noise(features)
            augmented = rotate_data(augmented)
            row = augmented.flatten().tolist() + [label]
            augmented_frames.append(row)
    df = pd.DataFrame(augmented_frames)
    with open(file_path, 'a', newline='') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)


def resample_frames1(gesture_data, target_frames=20):
    num_frames = len(gesture_data)
    if num_frames == target_frames:
        return gesture_data
    elif num_frames < target_frames:
        new_data = []
        indices = np.linspace(0, num_frames - 1, target_frames)
        for i in range(target_frames):
            index_floor = int(np.floor(indices[i]))
            index_ceil = min(int(np.ceil(indices[i])), num_frames - 1)
            weight = indices[i] - index_floor
            interpolated_frame = (1 - weight) * np.array(gesture_data[index_floor]) + weight * np.array(
                gesture_data[index_ceil])
            new_data.append(interpolated_frame.tolist())
        return new_data
    else:
        step = num_frames // target_frames
        new_data = [gesture_data[i * step] for i in range(target_frames)]
        return new_data


def resample_frames2(gesture_data, target_frames=20):
    num_frames = len(gesture_data)
    processed = [np.array(left + right) for left, right in gesture_data]
    if num_frames == target_frames:
        return processed
    elif num_frames < target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames)
        new_data = []
        for i in range(target_frames):
            idx_floor = int(np.floor(indices[i]))
            idx_ceil = min(int(np.ceil(indices[i])), num_frames - 1)
            w = indices[i] - idx_floor
            interpolated = (1 - w) * processed[idx_floor] + w * processed[idx_ceil]
            new_data.append(interpolated.tolist())
        return new_data
    else:
        step = num_frames / target_frames
        new_data = [processed[int(i * step)] for i in range(target_frames)]
        return new_data


if __name__ == "__main__":
    FILE_PATH = 'D:\Develop\Python\Code\Hands\data\dynamic_hand_dataset_f20.csv'
    # label = 1
    add_num = 100
