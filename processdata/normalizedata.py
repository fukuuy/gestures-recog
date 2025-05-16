import numpy as np


def normalize_hand_data1(hand_data):
    points = np.array(hand_data).reshape(21, 3)
    wrist = points[0]
    points -= wrist

    scale = np.linalg.norm(points[9] - points[0])
    points /= scale

    return points.flatten().tolist()


def normalize_hand_data(hand_data):
    if not hand_data:
        return []
    points = np.array(hand_data).reshape(21, 3)
    wrist = points[0]
    points -= wrist
    scale = np.linalg.norm(points[9] - points[0])
    points /= scale
    return points.flatten().tolist()


def normalize_hands_data(hands_data):
    left_hand = hands_data[0] if hands_data[0] else None
    right_hand = hands_data[1] if hands_data[1] else None
    if not (left_hand and right_hand):
        return [
            normalize_hand_data(left_hand) if left_hand else [],
            normalize_hand_data(right_hand) if right_hand else []
        ]
    left_points = np.array(left_hand).reshape(21, 3)
    right_points = np.array(right_hand).reshape(21, 3)
    all_points = np.vstack([left_points, right_points])
    center = np.mean(all_points, axis=0)
    left_points -= center
    right_points -= center

    left_palm_size = np.linalg.norm(left_points[9] - left_points[0])
    right_palm_size = np.linalg.norm(right_points[9] - right_points[0])
    interhand_dist = np.linalg.norm(left_points[0] - right_points[0])
    scale = (left_palm_size + right_palm_size + interhand_dist) / 3
    scale = max(scale, 1e-6)
    left_points /= scale
    right_points /= scale

    return [
        left_points.flatten().tolist(),
        right_points.flatten().tolist()
    ]
