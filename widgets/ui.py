import tkinter as tk
from tkinter import ttk


class GUI:
    def __init__(self):
        self.should_exit = False
        self.should_save = False

        self.root = tk.Tk()
        self.root.title("数据收集")
        control_frame = ttk.Frame(self.root, padding="5")
        control_frame.pack()

        label_frame = ttk.LabelFrame(control_frame, text="手势标签", padding="1")
        label_frame.pack(pady=5, fill=tk.X)
        self.label_entry = ttk.Entry(label_frame)
        self.label_entry.pack(pady=5)

        total_count_frame = ttk.LabelFrame(control_frame, text="收集数据量", padding="1")
        total_count_frame.pack(pady=5, fill=tk.X)
        self.total_count_entry = ttk.Entry(total_count_frame)
        self.total_count_entry.insert(0, "50")
        self.total_count_entry.pack(pady=5)

        delay_time_frame = ttk.LabelFrame(control_frame, text="开始延迟时间", padding="1")
        delay_time_frame.pack(pady=5, fill=tk.X)
        self.delay_time_entry = ttk.Entry(delay_time_frame)
        self.delay_time_entry.insert(0, "3")
        self.delay_time_entry.pack(pady=5)

        add_num_frame = ttk.LabelFrame(control_frame, text="增强数据数量", padding="1")
        add_num_frame.pack(pady=5, fill=tk.X)
        self.add_num_entry = ttk.Entry(add_num_frame)
        self.add_num_entry.insert(0, "0")
        self.add_num_entry.pack(pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5)
        self.save_button = ttk.Button(
            button_frame,
            text="开始保存",
            command=self.on_save,
            width=15
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.exit_button = ttk.Button(
            button_frame,
            text="退出",
            command=self.on_exit,
            width=15
        )
        self.exit_button.pack(side=tk.LEFT, padx=5)
        self.root.focus_force()

    def on_save(self):
        self.should_save = True

    def on_exit(self):
        self.should_exit = True
        self.root.destroy()

    def get_label(self):
        return self.label_entry.get()

    def get_total_count(self):
        return int(self.total_count_entry.get())

    def get_delay_time(self):
        return int(self.delay_time_entry.get())

    def get_add_num(self):
        return int(self.add_num_entry.get())

    def reset_save_flag(self):
        self.should_save = False

    def update(self):
        self.root.update_idletasks()
        self.root.update()


class DYGUI:
    def __init__(self):
        self.should_exit = False
        self.is_saving = False
        self.should_stop = False
        self.root = tk.Tk()

        self.root.title("数据收集")
        control_frame = ttk.Frame(self.root, padding="5")
        control_frame.pack()

        label_frame = ttk.LabelFrame(control_frame, text='手势标签', padding="1")
        label_frame.pack(pady=5, fill=tk.X)
        self.label_number = ttk.Entry(label_frame)
        self.label_number.pack(pady=5)

        add_num_frame = ttk.LabelFrame(control_frame, text="增强数据数量", padding="1")
        add_num_frame.pack(pady=5, fill=tk.X)
        self.add_num_entry = ttk.Entry(add_num_frame)
        self.add_num_entry.insert(0, "0")
        self.add_num_entry.pack(pady=5)
        self.root.focus_force()

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5)
        self.button = ttk.Button(button_frame, text="开始收集", command=self.on_save, width=15)
        self.button.pack(side=tk.LEFT, padx=5)

        self.exit_button = ttk.Button(button_frame, text="退出", command=self.on_exit, width=15)
        self.exit_button.pack(side=tk.LEFT, padx=5)

    def reset_save_flag(self):
        self.is_saving = False
        self.button.config(text="开始收集", command=self.on_save)

    def on_save(self):
        self.is_saving = True
        self.button.config(text="结束收集", command=self.on_stop)

    def on_stop(self):
        self.should_stop = True

    def on_exit(self):
        self.should_exit = True
        self.root.destroy()

    def get_label(self):
        return self.label_number.get()

    def update(self):
        self.root.update_idletasks()
        self.root.update()

    def get_add_num(self):
        return int(self.add_num_entry.get())


if __name__ == "__main__":
    gui = DYGUI()
    gui.root.mainloop()
