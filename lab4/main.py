import os
import shutil
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)
os.makedirs(plots_dir, exist_ok=True)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Bankloan.csv')
raw_df = pd.read_csv(data_path)

print("Исходная размерность датасета:", raw_df.shape)
print("Пропуски по столбцам:\n", raw_df.isna().sum())

df = raw_df.dropna(subset=['default']).copy()
df['default'] = df['default'].astype(int)
print("Размерность после удаления пропусков в целевой переменной:", df.shape)

feature_cols = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']
target_col = 'default'

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=target_col, palette='viridis')
plt.title('Распределение целевой переменной (default)')
plt.xlabel('Дефолт (0 - нет, 1 - да)')
plt.ylabel('Количество')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_target_distribution.png'), dpi=300)
plt.close()

plt.figure(figsize=(14, 8))
for i, col in enumerate(feature_cols, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col)
plt.suptitle('Ящики с усами до очистки от выбросов')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_features_boxplots_before.png'), dpi=300)
plt.close()

q1 = df[feature_cols].quantile(0.25)
q3 = df[feature_cols].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_mask = ((df[feature_cols] < lower_bound) | (df[feature_cols] > upper_bound)).any(axis=1)
df_clean = df[~outliers_mask].copy()

print("Количество обнаруженных выбросов (метод IQR):", outliers_mask.sum())
print("Размерность датасета после удаления выбросов:", df_clean.shape)
print("Распределение целевого признака после удаления выбросов:\n", df_clean[target_col].value_counts())

plt.figure(figsize=(14, 8))
for i, col in enumerate(feature_cols, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=df_clean[col], color='lightgreen')
    plt.title(col)
plt.suptitle('Ящики с усами после очистки от выбросов')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_features_boxplots_after.png'), dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
corr_matrix = df_clean[feature_cols + [target_col]].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_correlation_heatmap.png'), dpi=300)
plt.close()

X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), True),
    'SVM': (SVC(probability=True, random_state=42), True),
    'CART': (DecisionTreeClassifier(random_state=42, max_depth=4, criterion='gini'), False)
}

results = []
trained_models = {}
predictions = {}
probabilities = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, use_scaling) in models.items():
    X_tr = X_train_scaled if use_scaling else X_train
    X_te = X_test_scaled if use_scaling else X_test
    
    cv_res = cross_validate(
        model, X_tr, y_train, cv=cv, scoring=['accuracy', 'f1', 'roc_auc']
    )
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    trained_models[name] = model
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV Accuracy (mean)': cv_res['test_accuracy'].mean(),
        'CV F1 (mean)': cv_res['test_f1'].mean(),
        'CV ROC-AUC (mean)': cv_res['test_roc_auc'].mean()
    })
    
    print(f"\n==================== {name} ====================")
    print("Матрица ошибок (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("\nОтчет о классификации (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['No Default (0)', 'Default (1)']))

results_df = pd.DataFrame(results)
print("\nСводная таблица метрик качества моделей:")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{name}')
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_xticklabels(['No Default', 'Default'])
    ax.set_yticklabels(['No Default', 'Default'])
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_confusion_matrices.png'), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for name, y_prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые моделей')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '06_roc_curves.png'), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for name, y_prob in probabilities.items():
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall_vals, precision_vals, lw=2, label=name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривые моделей')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '07_precision_recall_curves.png'), dpi=300)
plt.close()

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
melted_df = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(data=melted_df, x='Metric', y='Score', hue='Model', palette='muted')
plt.title('Сравнение метрик качества моделей на тестовой выборке')
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '08_metrics_comparison.png'), dpi=300)
plt.close()

plt.figure(figsize=(18, 10))
plot_tree(
    trained_models['CART'],
    feature_names=feature_cols,
    class_names=['No Default', 'Default'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('Визуализация решающего дерева (CART)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '09_cart_decision_tree.png'), dpi=300)
plt.close()

cart_importances = pd.Series(trained_models['CART'].feature_importances_, index=feature_cols)
logreg_coefs = pd.Series(np.abs(trained_models['Logistic Regression'].coef_[0]), index=feature_cols)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cart_importances.sort_values().plot(kind='barh', ax=ax1, color='forestgreen')
ax1.set_title('Важность признаков в CART')
ax1.set_xlabel('Gini Importance')

logreg_coefs.sort_values().plot(kind='barh', ax=ax2, color='royalblue')
ax2.set_title('|Коэффициенты| в Logistic Regression')
ax2.set_xlabel('Абсолютное значение коэффициента')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '10_feature_importances.png'), dpi=300)
plt.close()

print(f"\nВсе графики успешно сохранены в папку: {plots_dir}")
