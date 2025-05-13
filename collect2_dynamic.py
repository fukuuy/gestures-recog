import csv
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from process.add_datas import save_augmented_frames
from process.normalizedata import normalize_hands_data
from widgets.ui import DYGUI
from widgets.showfigure import PLOT2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
plot2 = PLOT2()


def read_label():
    labels = set()
    try:
        with open(DATASET_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label = row[-1]
                labels.add(label)
        return sorted(labels)
    except FileNotFoundError:
        print(f"�ļ�δ����")
        return None


def save_data(data, hlabel):
    header = [f"{axis}0{i}" for i in range(0, 21) for axis in ["x", "y", "z"]] + [f"{axis}1{i}" for i in range(0, 21) for axis in ["x", "y", "z"]] + ["label"]
    save_frames = []
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        for i, features in enumerate(data):
            data_row = features[3:] + [hlabel]
            save_frames.append(data_row)
            writer.writerow(data_row)
    print(f"�ѱ������ݣ���ǩ��{hlabel}��")
    return save_frames


def resample_data(gesture_data, target_frames=20):
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


if __name__ == "__main__":
    tar_frame = 20
    DATASET_PATH = f"data/dynamic_hand2_dataset_f{tar_frame}.csv"

    frame = 0
    cap = cv2.VideoCapture(0)
    gui = DYGUI()
    plt.ion()
    sequence = []
    if read_label() is not None: print(f"�Ѽ�¼��ǩ{read_label()}")
    while cap.isOpened() and not gui.should_exit:
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(image, f"F_num: {frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('DynamicTwoHands', cv2.flip(image, 1))

        hands_data = [[], []]
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])
                hand_label = 0 if handedness.classification[0].label == 'Left' else 1
                hands_data[hand_label] = handxyz
            hands_data = normalize_hands_data(hands_data)
            plot2.update(hands_data)

        if gui.is_saving:
            label = int(gui.get_label())
            sequence.append(hands_data)
            add_num = gui.get_add_num()
            frame += 1
            if gui.should_stop:
                feature = resample_data(sequence, tar_frame)
                save_frames = save_data(feature, label)
                frame = 0
                gui.reset_save_flag()
                print("�ռ����")
                if add_num != 0:
                    save_augmented_frames(save_frames, add_num, DATASET_PATH)
                    print("������ǿ���")
        gui.update()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()
