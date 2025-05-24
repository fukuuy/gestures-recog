# Gesture Recognition

## 1. Project Overview
This project is a gesture recognition system that supports single-hand and double-hand static and dynamic video gesture collection, data enhancement, model training, and result display. By using Mediapipe to detect hand key points, combined with machine learning and deep learning models, and provides data enhancement methods for improving the generalization ability of the model.

## 2. Project Features
1. **Gesture Collection**: Supports single-hand and double-hand static and dynamic video gesture collection. Gesture labels can be specified through the graphical interface, and real-time feedback is provided during collection, with strong interactivity.
2. **Data Enhancement**: Provides multiple data enhancement methods, including rotation, flipping, etc., to improve the generalization ability of the model and expand the training data.
3. **Model Training**: Supports multiple machine learning models, such as KNN, XGBoost, etc., and deep learning models, such as LSTM for gesture classification.
4. **Result Display**: Provides 3D visualization of hand key points, facilitating intuitive observation of data.

## 3. Project Structure
```
gestures-recog/
    ├── collect1.py           # Single-hand static gesture collection
    ├── collect1_dynamic.py   # Single-hand dynamic gesture collection
    ├── collect2.py           # Double-hand static gesture collection
    ├── collect2_dynamic.py   # Double-hand dynamic gesture collection
    ├── processdata/              # Data processing module
    │   ├── add_datas.py      # Data enhancement implementation
    │   ├── normalizedata.py  # Data normalization
    ├── widgets/              # Graphical interface module
    │   ├── ui.py             # Interface implementation
    │   ├── showfigure.py     # 3D figure display
    ├── train.py              # Static gesture model training
    ├── train_dynamic.py      # Dynamic gesture model training
    ├── recognition.py        # Gesture recognition model
    ├── read_csv.py           # Static data reading and display
    ├── read_dynamic_csv.py   # Dynamic data reading and display
    ├── requirements.txt      # Project dependencies
    ├── data/                 # Dataset storage directory
        ├── hand_dataset.csv        # Single-hand static dataset
        ├── hand2_dataset.csv       # Double-hand static dataset
        ├── dynamic_hand_dataset_f20.csv    # Single-hand dynamic dataset
        ├── dynamic_hand2_dataset_f20.csv   # Double-hand dynamic dataset
```

## 4. Installation Steps
Execute the following command in the project directory to install the dependencies:
```bash
pip install -r requirements.txt
```

## 5. Usage Methods

### 1. Gesture Collection
- **Single-hand Static Gesture Collection**:
```bash
python collect1.py
```
- **Single-hand Dynamic Gesture Collection**:
```bash
python collect1_dynamic.py
```
- **Double-hand Static Gesture Collection**:
```bash
python collect2.py
```
- **Double-hand Dynamic Gesture Collection**:
```bash
python collect2_dynamic.py
```

### 2. Model Training
- **Static Gesture Model Training**:
```bash
python train.py
```
- **Dynamic Gesture Model Training**:
```bash
python train_dynamic.py
```

### 3. Model Prediction
- **Static Gesture Recognition**:
```bash
python recognition.py
```
- **Dynamic Gesture Recognition**:
```bash
python recognition_dynamic.py
```
- **Static Single-hand Gesture Recognition**:
```bash
python recognition1.py
```
- **Static Double-hand Gesture Recognition**:
```bash
python recognition2.py
```
- **Dynamic Single-hand Gesture Recognition**:
```bash
python recognition1_dynamic.py
```
- **Dynamic Double-hand Gesture Recognition**:
```bash
python recognition2_dynamic.py
```

### 3. Data Visualization (for viewing the dataset)
- **Static Data Visualization**:
```bash
python read_csv.py
```
- **Dynamic Data Visualization**:
```bash
python read_dynamic_csv.py
```

## 6. Related Notes
- `requirements.txt` specifies the Python library versions required for the project.
- In `train.py` and `train_dynamic.py`, you can modify the following parameters:
  - `RANDOM_STATE`: Random seed, used to ensure the reproducibility of experimental results.
  - `EPOCHS`: Number of training epochs.
  - `BATCH_SIZE`: Batch size.
  - `FILE_PATH`: Dataset file path.

## 7. Precautions
- Ensure that the camera device is working properly before collecting gestures.
- Data enhancement should be selected according to the actual situation to avoid overfitting.
- When training the model, consider the size of the dataset and the complexity of the model to avoid resource waste.

## 8. Contribute

Welcome to contribute to this project, you can participate by submitting an issue or pull request.
