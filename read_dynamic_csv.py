import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv


def read_csv_line(file_path, start_line, tar_frame):
    frames = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for _ in range(start_line * 20):
            next(reader, None)
        for i in range(tar_frame):
            row = next(reader, None)
            frame = [0, 0, 0] + [float(val) for val in row[:-1]]
            frames.append(frame)
        return frames


fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (13, 17), (0, 17)
]

lines = [ax.plot([], [], [], c='r', linestyle='-')[0] for _ in range(21)]
scatter = ax.scatter([], [], [], c='b', marker='o')
for line in lines:
    line.set_data([], [])
    line.set_3d_properties([])
scatter._offsets3d = (np.array([]), np.array([]), np.array([]))


def update(hand_data):
    handx = -1 * np.array(hand_data[0::3])
    handy = np.array(hand_data[2::3])
    handz = -1 * np.array(hand_data[1::3])
    ax.set_xlim(min(handx) - 0.1, max(handx) + 0.1)
    ax.set_ylim(min(handy) - 0.1, max(handy) + 0.1)
    ax.set_zlim(min(handz) - 0.1, max(handz) + 0.1)
    for j, (start, end) in enumerate(connections):
        lines[j].set_data([handx[start], handx[end]], [handy[start], handy[end]])
        lines[j].set_3d_properties([handz[start], handz[end]])
    scatter._offsets3d = (handx, handy, handz)
    return lines + [scatter]


if __name__ == '__main__':
    TAR_FRAME = 20
    FILE_PATH = f"data/dynamic_hand_dataset_f{TAR_FRAME}.csv"
    LINE_NUM = 300

    frames = read_csv_line(FILE_PATH, LINE_NUM - 1, TAR_FRAME)
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=200,
        repeat=False,
    )
    plt.show()
