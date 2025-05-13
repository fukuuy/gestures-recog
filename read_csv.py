import csv
import matplotlib.pyplot as plt
import numpy as np
# from add_datas import rotate_data, add_random_noise


def read_csv_line(file_path, start_line):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for i, row in enumerate(reader, start=1):
                if i == start_line:
                    return row
    except FileNotFoundError:
        print(f"未找到文件 {file_path}。")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None


def show_1_plot(hand_data):
    plt.figure(figsize=(5, 4))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    lines = []
    scatters = []
    hand_lines = [ax.plot([], [], [], c='r', linestyle='-')[0] for _ in range(21)]
    lines.append(hand_lines)
    scatter = ax.scatter([], [], [], c='b', marker='o')
    scatters.append(scatter)
    for hand_lines in lines:
        for line in hand_lines:
            line.set_data([], [])
            line.set_3d_properties([])

    for scatter in scatters:
        scatter._offsets3d = (np.array([]), np.array([]), np.array([]))

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (13, 17), (0, 17),
    ]

    handx = -1 * np.array(hand_data[0::3])
    handy = np.array(hand_data[2::3])
    handz = -1 * np.array(hand_data[1::3])

    for j, (start, end) in enumerate(connections):
        lines[0][j].set_data([handx[start], handx[end]], [handy[start], handy[end]])
        lines[0][j].set_3d_properties([handz[start], handz[end]])
    scatters[0]._offsets3d = (handx, handy, handz)

    if hand_data:
        ax.set_xlim(min(handx) - 0.1, max(handx) + 0.1)
        ax.set_ylim(min(handy) - 0.1, max(handy) + 0.1)
        ax.set_zlim(min(handz) - 0.1, max(handz) + 0.1)


def show_2_plot(hands_data):
    plt.figure(figsize=(5, 4))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    hand_colors = ['red', 'blue']
    lines = []
    scatters = []
    for color in hand_colors:
        hand_lines = [ax.plot([], [], [], c=color, linestyle='-')[0] for _ in range(21)]
        lines.append(hand_lines)
        scatter = ax.scatter([], [], [], c=color, marker='o')
        scatters.append(scatter)
        for hand_lines in lines:
            for line in hand_lines:
                line.set_data([], [])
                line.set_3d_properties([])

    for scatter in scatters:
        scatter._offsets3d = (np.array([]), np.array([]), np.array([]))

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (13, 17), (0, 17),
    ]
    all_x, all_y, all_z = [], [], []
    hands_data = [hands_data[:63], hands_data[63:]]
    for hand_idx, hand_data in enumerate(hands_data):
        if hand_data:
            handxyz = np.array(hand_data).reshape(21, 3)
            handx = -handxyz[:, 0]
            handy = handxyz[:, 2]
            handz = -handxyz[:, 1]
            color = hand_colors[hand_idx]
            for j, (start, end) in enumerate(connections):
                lines[hand_idx][j].set_data([handx[start], handx[end]], [handy[start], handy[end]])
                lines[hand_idx][j].set_3d_properties([handz[start], handz[end]])
                lines[hand_idx][j].set_color(color)
            scatters[hand_idx]._offsets3d = (handx, handy, handz)
            all_x.extend(handx)
            all_y.extend(handy)
            all_z.extend(handz)
    if all_x:
        ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
        ax.set_ylim(min(all_y) - 0.1, max(all_y) + 0.1)
        ax.set_zlim(min(all_z) - 0.1, max(all_z) + 0.1)


if __name__ == "__main__":
    FILE_PATH = 'data/hand1_dataset.csv'
    LINE_NUM = 1000

    read_data = read_csv_line(FILE_PATH, LINE_NUM)
    if FILE_PATH == 'data/hand_dataset.csv':
        if read_data is not None:
            read_data = np.array(read_data, dtype=np.float32)
            hand_data = [0, 0, 0] + read_data[:-1].tolist()
            show_1_plot(hand_data)

    elif FILE_PATH == 'data/hand2_dataset.csv':
        if read_data is not None:
            read_data = np.array(read_data, dtype=np.float32)
            hand_data = read_data[:-1].tolist()
            show_2_plot(hand_data)

    plt.show()
