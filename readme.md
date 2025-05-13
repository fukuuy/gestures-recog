# 手势识别项目

## 一、项目概述
本项目是一个基于Python的手势识别系统，利用Mediapipe库进行手部关键点检测，并结合多种机器学习模型（如随机森林、支持向量机、K近邻、XGBoost和多层感知机）实现手势的分类识别。项目支持单手势和双手势的识别，同时提供了数据收集、数据增强、模型训练和实时识别等功能。

## 二、项目结构
```
Gestrue_Recognition/
├── add_datas.py            # 数据增强脚本
├── collect.py              # 单手势数据收集脚本
├── collect2.py             # 双手势数据收集脚本
├── normalizedata.py        # 数据归一化处理脚本
├── recognition.py          # 单双手势实时识别脚本
├── recognition1.py         # 单手势实时识别脚本
├── recognition2.py         # 双手势实时识别脚本
├── requirements.txt        # 项目依赖库文件
├── train.py                # 模型训练脚本
├── ui.py                   # 数据收集界面脚本
├── read_csv.py             # 读取CSV文件并可视化手势数据脚本
└── data/                   # 数据集存储目录
    ├── hand_dataset.csv    # 单手势数据集
    └── hand2_dataset.csv   # 双手势数据集
└── weight/                 # 模型权重和标签映射存储目录
    ├── hand/               # 单手势模型相关文件
    │   ├── best_model.pkl  # 单手势最佳模型
    │   └── label_encoder.pkl # 单手势标签映射
    └── hands/              # 双手势模型相关文件
        ├── best_model.pkl  # 双手势最佳模型
        └── label_encoder.pkl # 双手势标签映射
```

## 三、环境配置
```bash
pip install -r requirements.txt
```

## 四、使用方法

### 1. 数据收集
#### 单手势数据收集
```bash
python collect1.py
```
运行该脚本后，会弹出一个图形界面，你可以设置手势标签、收集数据量、延迟时间和增强数据数量。点击“开始保存”按钮，程序会在延迟指定时间后开始收集数据，并将数据保存到`data/hand_dataset.csv`文件中。收集完成后，若设置了增强数据数量，程序会自动进行数据增强。

#### 双手势数据收集
```bash
python collect2.py
```
使用方法与单手势数据收集类似，数据会保存到`data/hand2_dataset.csv`文件中。

### 2. 模型训练
```bash
python train.py
```
该脚本会加载指定的数据集（默认是`data/hand2_dataset.csv`），进行数据预处理、模型训练和评估，并选择最佳模型保存到`weight`目录下。你可以根据需要修改`train.py`中的`FILE_PATH`变量来选择不同的数据集，以及修改SINGLE_LABELS或DOUBLE_LABELS变量来配置标签的映射。

### 3. 实时识别
#### 单手势实时识别
```bash
python recognition1.py
```
程序会打开摄像头，实时识别单手势，并在图像上显示识别结果。

#### 双手势实时识别
```bash
python recognition2.py
```
程序会打开摄像头，实时识别双手势，并在图像上显示识别结果。

#### 单双手势实时识别
```bash
python recognition.py
```
该脚本可以同时处理单手势和双手势的识别，并根据检测到的手部数量自动选择合适的模型进行预测。

### 4. 数据可视化
```bash
python read_csv.py
```
该脚本可以读取指定CSV文件中的手势数据，并将其可视化展示。你可以修改`read_csv.py`中的`FILE_PATH`和`LINE_NUMBER`变量来选择不同的数据进行可视化。

## 五、注意事项
- 确保摄像头正常工作，并且在光线充足的环境下进行数据收集和识别。
- 数据增强的参数（如噪声比例、旋转角度范围等）可以在`add_datas.py`文件中进行调整。
- 模型训练的参数（如模型类型、超参数搜索范围等）可以在`train.py`文件中进行修改。
