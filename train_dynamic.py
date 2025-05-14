import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    dim = X.shape[1]
    sequences_num = len(X) // SEQUENCE_LEN
    X = X[:sequences_num * SEQUENCE_LEN].reshape(-1, SEQUENCE_LEN, dim)
    y = y[:sequences_num * SEQUENCE_LEN].reshape(-1, SEQUENCE_LEN)
    y = y[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print("训练集类别分布:", np.unique(np.argmax(y_train, axis=1), return_counts=True))
    print("测试集类别分布:", np.unique(np.argmax(y_test, axis=1), return_counts=True))
    return X_train, X_test, y_train, y_test


def train_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy * 100:.2f}%")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SEQUENCE_LEN = 20
    RANDOM_STATE = 30
    EPOCHS = 5
    BATCH_SIZE = 16
    FILE_PATH = f"data/dynamic_hand2_dataset_f{SEQUENCE_LEN}.csv"
    SINGLE_LABELS = {
        'eat': [0],
        'circle': [1],
        'come on': [2],
    }
    DOUBLE_LABELS = {
        'one': [0],
        'two': [1],
        'three': [2],
    }

    if FILE_PATH == f'data/dynamic_hand_dataset_f{SEQUENCE_LEN}.csv':
        dictionary = 'hand'
        LABELS = SINGLE_LABELS

    elif FILE_PATH == f'data/dynamic_hand2_dataset_f{SEQUENCE_LEN}.csv':
        dictionary = 'hands'
        LABELS = DOUBLE_LABELS
    else:
        print("文件名错误")
        exit(0)
    joblib.dump(LABELS, f'models/{dictionary}/dy_label_encoder.pkl')
    print("标签映射已保存为 dy_label_encoder.pkl")

    X_train, X_test, y_train, y_test = load_data(FILE_PATH)
    input_shape = (SEQUENCE_LEN, X_train.shape[2])
    model, history = train_model(input_shape, y_train.shape[1])

    evaluate_model(model, X_test, y_test)

    model.save(f'models/{dictionary}/dy_model.h5')
    print("模型保存为 dy_model.h5")
