import csv
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from process.add_datas import save_augmented_data
from process.normalizedata import normalize_hands_data
from widgets.ui import GUI
import time
from widgets.showfigure import PLOT2

DATASET_FILE = "data/hand2_dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
plot2 = PLOT2()


def read_label(file_path=DATASET_FILE):
    labels = set()
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label = row[-1]
                labels.add(label)
    except FileNotFoundError:
        print(f"文件未创建")
    return sorted(labels)


def save_data(features, hlabel):
    header = [f"{axis}0{i}" for i in range(0, 21) for axis in ["x", "y", "z"]] + [f"{axis}1{i}" for i in range(0, 21) for axis in ["x", "y", "z"]] + ["label"]
    data_row = np.append(features[0], features[1]).tolist() + [hlabel]
    with open(DATASET_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        writer.writerow(data_row)
    print(f"已保存（标签：{hlabel}）")


if __name__ == "__main__":
    collecting = False
    start_time = None
    plt.ion()
    cap = cv2.VideoCapture(0)
    gui = GUI()
    count = 0
    print(f"已记录标签{read_label()}")
    while cap.isOpened() and not gui.should_exit:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('TwoHands', cv2.flip(image, 1))

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

        if gui.should_save:
            start_time = time.time()
            collecting = True
            total_count = gui.get_total_count()
            delay_time = gui.get_delay_time()
            add_num = gui.get_add_num()
            gui.reset_save_flag()

        if collecting:
            passed_time = time.time() - start_time
            label = int(gui.get_label())
            if passed_time >= delay_time and count < total_count:
                save_data(hands_data, label)
                count += 1
                print(f'已收集{count}条数据')
            elif count >= total_count:
                collecting = False
                count = 0
                print("收集完成")
                if add_num != 0:
                    save_augmented_data(label, add_num, DATASET_FILE)
                    print("数据增强完成")
            else:
                print(f'{int(delay_time - passed_time)}秒后开始记录')
        gui.update()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()
