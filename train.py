from time import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(how='any')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print("训练集类别分布:", np.unique(y_train, return_counts=True))
    print("测试集类别分布:", np.unique(y_test, return_counts=True))
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type):
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5]
        }
        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        base_model = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
    elif model_type == 'knn':
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
        base_model = KNeighborsClassifier(n_jobs=-1)
    elif model_type == 'xgboost':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'num_class': [len(np.unique(y_train))]
        }
        base_model = XGBClassifier(
            objective='multi:softmax',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss'
        )
    elif model_type == 'mlp':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001]
        }
        base_model = MLPClassifier(
            solver='adam',
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True
        )
    else:
        raise ValueError("未知的模型类型")

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # 交叉验证
        model = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)

        print(f"最佳参数: {model.best_params_}")
        return model.best_estimator_
    except Exception as e:
        print(f"训练{model_type}模型时出错: {str(e)}")
        base_model.fit(X_train, y_train)
        return base_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n分类报告:")
    print(report)
    return accuracy, f1


if __name__ == "__main__":
    RANDOM_STATE = 30
    FILE_PATH = 'data/hand2_dataset.csv'
    SINGLE_LABELS = {
        'one': [0],
        'two': [1, 7],
        'three': [2, 8],
        'four': [3],
        'five': [4],
        'six': [5, 6],
    }
    DOUBLE_LABELS = {
        'love': [0],
        'good': [1],
        'ten': [2],
    }

    if FILE_PATH == 'data/hand_dataset.csv':
        dictionary = 'hand'
        LABELS = SINGLE_LABELS
    elif FILE_PATH == 'data/hand2_dataset.csv':
        dictionary = 'hands'
        LABELS = DOUBLE_LABELS
    else:
        print("文件名错误")
        exit(0)
    models = {
        'random_forest': "随机森林",
        'svm': "支持向量机",
        'knn': "K近邻",
        'xgboost': "XGBoost",
        'mlp': "多层感知机"
    }
    X_train, X_test, y_train, y_test = load_data(FILE_PATH)
    joblib.dump(LABELS, f'models/{dictionary}/label_encoder.pkl')
    print("标签映射已保存为label_encoder.pkl")
    best_model = None
    best_accuracy = 0
    best_f1 = 0
    for model_type, name in models.items():
        print(f"\n训练 {name} 模型")
        start = time()
        model = train_model(X_train, y_train, model_type)
        accuracy, f1 = evaluate_model(model, X_test, y_test)

        if f1 > best_f1 or (f1 == best_f1 and accuracy > best_accuracy):
            best_accuracy = accuracy
            best_f1 = f1
            best_model = model
            print(f"新的最佳模型: {name}")

        end = time()
        t = end - start
        print("训练时间：%.3fs" % t)

    if best_model is not None:
        joblib.dump(best_model, f'models/{dictionary}/best_model.pkl')
        print("\n最佳模型已保存为'best_model.pkl'")
