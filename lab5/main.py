import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

output_dir = "plots"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

class Node:
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index
        self.t = t
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()

    def predict(self):
        classes = {}
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        prediction = max(classes, key=classes.get)
        return prediction

def gini(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1.0 - np.sum(probabilities ** 2)

def gain(left_labels, right_labels, root_gini):
    total_len = len(left_labels) + len(right_labels)
    if total_len == 0 or len(left_labels) == 0 or len(right_labels) == 0:
        return 0.0
    p_left = len(left_labels) / total_len
    p_right = len(right_labels) / total_len
    return root_gini - p_left * gini(left_labels) - p_right * gini(right_labels)

def split(data, labels, column_index, t):
    left = np.where(data[:, column_index] <= t)
    right = np.where(data[:, column_index] > t)
    true_data = data[left]
    false_data = data[right]
    true_labels = labels[left]
    false_labels = labels[right]
    return true_data, false_data, true_labels, false_labels

def find_best_split(data, labels, min_samples_leaf=3):
    root_gini = gini(labels)
    best_gain = 0.0
    best_t = None
    best_index = None
    n_features = data.shape[1]
    for index in range(n_features):
        feature_values = np.unique(data[:, index])
        if len(feature_values) < 2:
            continue
        thresholds = (feature_values[:-1] + feature_values[1:]) / 2.0
        for t in thresholds:
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
            if len(true_labels) < min_samples_leaf or len(false_labels) < min_samples_leaf:
                continue
            current_gain = gain(true_labels, false_labels, root_gini)
            if current_gain > best_gain:
                best_gain, best_t, best_index = current_gain, t, index
    return best_gain, best_t, best_index

def build_tree(data, labels, max_depth=3, min_samples_split=2, min_samples_leaf=3, depth=0):
    if max_depth is not None and depth >= max_depth:
        return Leaf(data, labels)
    if len(labels) < min_samples_split:
        return Leaf(data, labels)
    best_gain, t, index = find_best_split(data, labels, min_samples_leaf)
    if best_gain <= 0.0 or t is None:
        return Leaf(data, labels)
    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
    true_branch = build_tree(true_data, true_labels, max_depth, min_samples_split, min_samples_leaf, depth + 1)
    false_branch = build_tree(false_data, false_labels, max_depth, min_samples_split, min_samples_leaf, depth + 1)
    return Node(index, t, true_branch, false_branch)

def classify_object(obj, node):
    if isinstance(node, Leaf):
        return node.prediction
    if obj[node.index] <= node.t:
        return classify_object(obj, node.true_branch)
    else:
        return classify_object(obj, node.false_branch)

def predict(data, tree):
    return [classify_object(obj, tree) for obj in data]

def print_tree(node, feature_names=None, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Прогноз:", node.prediction)
        return
    feature_name = f"Индекс {node.index}" if feature_names is None else feature_names[node.index]
    print(spacing + f"{feature_name} <= {node.t:.4f}")
    print(spacing + "--> True:")
    print_tree(node.true_branch, feature_names, spacing + "  ")
    print(spacing + "--> False:")
    print_tree(node.false_branch, feature_names, spacing + "  ")

def accuracy_metric(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(actual == predicted)

def precision_metric(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_metric(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    tp = np.sum((actual == 1) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_metric(actual, predicted):
    p = precision_metric(actual, predicted)
    r = recall_metric(actual, predicted)
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

def confusion_matrix_metric(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    tp = np.sum((actual == 1) & (predicted == 1))
    return np.array([[tn, fp], [fn, tp]])

data_path = "Bankloan.csv"
dataframe = pd.read_csv(data_path)
dataframe = dataframe.dropna(subset=["default"])

feature_columns = [col for col in dataframe.columns if col != "default"]
target_column = "default"

X = dataframe[feature_columns].values
y = dataframe[target_column].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

max_depth = 3
min_samples_split = 2
min_samples_leaf = 3

my_tree = build_tree(
    X_train,
    y_train,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf
)

sklearn_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
sklearn_tree.fit(X_train, y_train)

print("Структура самописного дерева решений:")
print_tree(my_tree, feature_names=feature_columns)

custom_train_preds = predict(X_train, my_tree)
custom_test_preds = predict(X_test, my_tree)

sklearn_train_preds = sklearn_tree.predict(X_train)
sklearn_test_preds = sklearn_tree.predict(X_test)

train_match = np.all(custom_train_preds == sklearn_train_preds)
test_match = np.all(custom_test_preds == sklearn_test_preds)

print("\nРезультаты сравнения:")
print(f"Полное совпадение на обучающей выборке: {train_match}")
print(f"Полное совпадение на тестовой выборке: {test_match}")

custom_acc = accuracy_metric(y_test, custom_test_preds)
custom_prec = precision_metric(y_test, custom_test_preds)
custom_rec = recall_metric(y_test, custom_test_preds)
custom_f1 = f1_metric(y_test, custom_test_preds)

sklearn_acc = accuracy_metric(y_test, sklearn_test_preds)
sklearn_prec = precision_metric(y_test, sklearn_test_preds)
sklearn_rec = recall_metric(y_test, sklearn_test_preds)
sklearn_f1 = f1_metric(y_test, sklearn_test_preds)

metrics_df = pd.DataFrame({
    "Метрика": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Самописное дерево": [custom_acc, custom_prec, custom_rec, custom_f1],
    "Sklearn дерево": [sklearn_acc, sklearn_prec, sklearn_rec, sklearn_f1]
})

print("\nМетрики моделей на тестовой выборке:")
print(metrics_df.to_string(index=False))

plt.figure(figsize=(8, 5))
bar_width = 0.35
indices = np.arange(len(metrics_df))
plt.bar(indices - bar_width / 2, metrics_df["Самописное дерево"], width=bar_width, label="Custom Tree", color="steelblue")
plt.bar(indices + bar_width / 2, metrics_df["Sklearn дерево"], width=bar_width, label="Sklearn Tree", color="forestgreen")
plt.xticks(indices, metrics_df["Метрика"], fontsize=11)
plt.ylabel("Значение", fontsize=11)
plt.ylim(0, 1.1)
plt.title("Сравнение метрик качества классификации", fontsize=13)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300)
plt.close()

custom_cm = confusion_matrix_metric(y_test, custom_test_preds)
sklearn_cm = confusion_matrix_metric(y_test, sklearn_test_preds)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(custom_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title("Матрица ошибок (Custom Tree)")
axes[0].set_xlabel("Предсказанный класс")
axes[0].set_ylabel("Истинный класс")

sns.heatmap(sklearn_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
axes[1].set_title("Матрица ошибок (Sklearn Tree)")
axes[1].set_xlabel("Предсказанный класс")
axes[1].set_ylabel("Истинный класс")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
plt.close()

depths = range(1, 9)
custom_depth_acc = []
sklearn_depth_acc = []

for d in depths:
    t_custom = build_tree(X_train, y_train, max_depth=d, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    t_sk = DecisionTreeClassifier(criterion="gini", max_depth=d, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
    t_sk.fit(X_train, y_train)
    custom_depth_acc.append(accuracy_metric(y_test, predict(X_test, t_custom)))
    sklearn_depth_acc.append(accuracy_metric(y_test, t_sk.predict(X_test)))

plt.figure(figsize=(9, 5))
plt.plot(depths, custom_depth_acc, marker="o", linewidth=2.5, label="Custom Tree", color="steelblue")
plt.plot(depths, sklearn_depth_acc, marker="x", linestyle="--", linewidth=2, label="Sklearn Tree", color="crimson")
plt.xlabel("Максимальная глубина (max_depth)", fontsize=11)
plt.ylabel("Точность (Accuracy)", fontsize=11)
plt.title("Зависимость точности от максимальной глубины дерева", fontsize=13)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_vs_depth.png"), dpi=300)
plt.close()

importances = sklearn_tree.feature_importances_
feature_order = np.argsort(importances)

plt.figure(figsize=(9, 5))
plt.barh(range(len(feature_order)), importances[feature_order], color="teal", align="center")
plt.yticks(range(len(feature_order)), [feature_columns[i] for i in feature_order], fontsize=10)
plt.xlabel("Важность признака (Gini Importance)", fontsize=11)
plt.title("Важность признаков для классификации неплательщиков", fontsize=13)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=300)
plt.close()

plt.figure(figsize=(16, 10))
plot_tree(
    sklearn_tree,
    feature_names=feature_columns,
    class_names=["No Default", "Default"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Визуализация дерева решений", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "decision_tree_structure.png"), dpi=300)
plt.close()

print(f"\nВсе графики успешно сгенерированы и сохранены в папку: {output_dir}")
