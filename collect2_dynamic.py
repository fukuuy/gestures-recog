import csv
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from process.add_datas import save_augmented_frames, resample_frames2
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
        print(f"文件未创建")
        return None


def save_data(data, label):
    header = [f"{axis}0{i}" for i in range(0, 21) for axis in ["x", "y", "z"]] + [f"{axis}1{i}" for i in range(0, 21) for axis in ["x", "y", "z"]] + ["label"]
    save_frames = []
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        for features in data:
            data_row = features.tolist() + [label]
            save_frames.append(data_row)
            writer.writerow(data_row)
    print(f"已保存数据（标签：{label}）")
    return save_frames


if __name__ == "__main__":
    tar_frame = 20
    DATASET_PATH = f"data/dynamic_hand2_dataset_f{tar_frame}.csv"

    frame = 0
    cap = cv2.VideoCapture(0)
    gui = DYGUI()
    plt.ion()
    sequence = []
    if read_label() is not None: print(f"已记录标签{read_label()}")
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
                frame = 0
                gui.reset_save_flag()
                gui.should_stop = False
                feature = resample_frames2(sequence, tar_frame)
                save_frames = save_data(feature, label)
                print("收集完成")
                if add_num != 0:
                    save_augmented_frames(save_frames, add_num, DATASET_PATH)
                    print("数据增强完成")
        gui.update()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()
