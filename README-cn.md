# 手势识别项目

## 一、项目简介
本项目是一个手势识别系统，支持单双手静态和动态手势的数据收集、模型训练以及可视化展示。通过使用Mediapipe库进行手部关键点检测，结合多种机器学习和深度学习模型进行手势分类，同时提供了数据增强功能以提高模型的泛化能力。

## 二、功能特点
1. **数据收集**：支持单双手静态和动态手势的数据收集，可通过图形界面指定手势标签、收集数据量、延迟时间和数据增强数量。
2. **数据增强**：提供数据平移、添加噪声和旋转等增强方法，可增加训练数据的多样性。
3. **模型训练**：支持多种机器学习模型（如随机森林、支持向量机、K近邻、XGBoost、多层感知机）和深度学习模型（LSTM）进行手势分类。
4. **可视化展示**：提供3D可视化展示手部关键点，可直观观察手势数据。

## 三、项目结构
```
gestures-recog/
├── collect1.py           # 单手静态数据收集
├── collect1_dynamic.py   # 单手动态数据收集
├── collect2.py           # 双手静态数据收集
├── collect2_dynamic.py   # 双手动态数据收集
├── process/              # 数据处理模块
│   ├── add_datas.py      # 数据增强和重采样
│   └── normalizedata.py  # 数据归一化
├── widgets/              # 图形界面模块
│   ├── ui.py             # 控制界面实现
│   └── showfigure.py     # 3D图形展示
├── train.py              # 静态手势模型训练
├── train_dynamic.py      # 动态手势模型训练
├── recognition.py        # 手势识别模块
├── read_csv.py           # 静态数据读取和可视化
├── read_dynamic_csv.py   # 动态数据读取和可视化
├── requirements.txt      # 项目依赖
└── data/                 # 数据集存储目录
    ├── hand_dataset.csv        # 单手静态数据集
    ├── hand2_dataset.csv       # 双手静态数据集
    ├── dynamic_hand_dataset_f20.csv    # 单手动态数据集
    └── dynamic_hand2_dataset_f20.csv   # 双手动态数据集
```

## 四、安装依赖
在项目根目录下执行以下命令安装所需依赖：
```bash
pip install -r requirements.txt
```

## 五、使用方法

### 1.数据收集
- **单手静态数据收集**：
```bash
python collect1.py
```
- **单手动态数据收集**：
```bash
python collect1_dynamic.py
```
- **双手静态数据收集**：
```bash
python collect2.py
```
- **双手动态数据收集**：
```bash
python collect2_dynamic.py
```

### 2.模型训练
- **静态手势模型训练**：
```bash
python train.py
```
- **动态手势模型训练**：
```bash
python train_dynamic.py
```
### 3.模型运行
- **静态手势识别**：
```bash
python recognition.py
```

- **动态手势识别**：
```bash
python recognition_dynamic.py
```

- **静态仅单手手势识别**：
```bash
python recognition1.py
```

- **静态仅双手手势识别**：
```bash
python recognition2.py
```

- **动态仅单手手势识别**：
```bash
python recognition1_dynamic.py
```

- **动态仅双手手势识别**：
```bash
python recognition2_dynamic.py
```

### 3.数据可视化(用于查看数据集)
- **静态数据可视化**：
```bash
python read_csv.py
```
- **动态数据可视化**：
```bash
python read_dynamic_csv.py
```


## 六、特别说明
- `requirements.txt`：指定了项目所需的Python库及其版本。
- `train.py` 和 `train_dynamic.py` 中可修改以下参数：
  - `RANDOM_STATE`：随机种子，用于保证实验结果的可重复性。
  - `EPOCHS`：训练轮数。
  - `BATCH_SIZE`：批次大小。
  - `FILE_PATH`：数据集文件路径。

## 七、注意事项
- 确保摄像头设备正常工作，以便进行数据收集。
- 数据增强数量可根据实际情况调整，以避免过拟合。
- 模型训练时间可能较长，取决于数据集大小和模型复杂度。
