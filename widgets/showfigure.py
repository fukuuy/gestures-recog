import matplotlib.pyplot as plt
import numpy as np


class PLOT1:
    def __init__(self):
        plt.figure(figsize=(5, 4))
        self.ax = plt.subplot(projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.lines = [self.ax.plot([], [], [], c='r', linestyle='-')[0] for _ in range(21)]
        self.scatters = self.ax.scatter([], [], [], c='b', marker='o')

    def update(self, hand_data):
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])

        self.scatters._offsets3d = (np.array([]), np.array([]), np.array([]))

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (13, 17), (0, 17)
        ]

        handx = -1 * np.array(hand_data[0::3])
        handy = np.array(hand_data[2::3])
        handz = -1 * np.array(hand_data[1::3])

        for j, (start, end) in enumerate(connections):
            self.lines[j].set_data([handx[start], handx[end]], [handy[start], handy[end]])
            self.lines[j].set_3d_properties([handz[start], handz[end]])
        self.scatters._offsets3d = (handx, handy, handz)

        if hand_data:
            self.ax.set_xlim(min(handx) - 0.1, max(handx) + 0.1)
            self.ax.set_ylim(min(handy) - 0.1, max(handy) + 0.1)
            self.ax.set_zlim(min(handz) - 0.1, max(handz) + 0.1)

        plt.draw()
        plt.pause(0.001)


class PLOT2:
    def __init__(self):
        plt.figure(figsize=(5, 4))
        self.ax = plt.subplot(projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.hand_colors = ['red', 'blue']
        self.lines = []
        self.scatters = []
        for color in self.hand_colors:
            hand_lines = [self.ax.plot([], [], [], c=color, linestyle='-')[0] for _ in range(21)]
            self.lines.append(hand_lines)
            scatter = self.ax.scatter([], [], [], c=color, marker='o')
            self.scatters.append(scatter)

    def update(self, hands_data):
        for hand_lines in self.lines:
            for line in hand_lines:
                line.set_data([], [])
                line.set_3d_properties([])

        for scatter in self.scatters:
            scatter._offsets3d = (np.array([]), np.array([]), np.array([]))

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (13, 17), (0, 17)
        ]
        all_x, all_y, all_z = [], [], []
        for hand_idx, hand_data in enumerate(hands_data):
            if hand_data:
                handxyz = np.array(hand_data).reshape(21, 3)
                handx = -handxyz[:, 0]
                handy = handxyz[:, 2]
                handz = -handxyz[:, 1]
                color = self.hand_colors[hand_idx]
                for j, (start, end) in enumerate(connections):
                    self.lines[hand_idx][j].set_data([handx[start], handx[end]], [handy[start], handy[end]])
                    self.lines[hand_idx][j].set_3d_properties([handz[start], handz[end]])
                    self.lines[hand_idx][j].set_color(color)
                self.scatters[hand_idx]._offsets3d = (handx, handy, handz)
                all_x.extend(handx)
                all_y.extend(handy)
                all_z.extend(handz)
        if all_x:
            self.ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
            self.ax.set_ylim(min(all_y) - 0.1, max(all_y) + 0.1)
            self.ax.set_zlim(min(all_z) - 0.1, max(all_z) + 0.1)

        plt.draw()
        plt.pause(0.001)

