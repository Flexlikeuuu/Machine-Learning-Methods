import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

plots_dir = 'plots'
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv('CarPrice_Assignment.csv')

df = df.drop(columns=['car_ID'])

df['Company'] = df['CarName'].apply(lambda x: x.split()[0].lower())
df = df.drop(columns=['CarName'])

brand_corrections = {
    'maxda': 'mazda',
    'porcshce': 'porsche',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen',
    'alfa-romero': 'alfa-romeo'
}
df['Company'] = df['Company'].replace(brand_corrections)

door_map = {'two': 2, 'four': 4}
cylinder_map = {
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'eight': 8,
    'twelve': 12
}

df['doornumber'] = df['doornumber'].map(door_map)
df['cylindernumber'] = df['cylindernumber'].map(cylinder_map)

categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df = df.dropna()

X = df.drop(columns=['price'])
y = df['price']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_model = GradientBoostingRegressor(random_state=42)
base_model.fit(X_train_scaled, y_train)

y_pred_base = base_model.predict(X_test_scaled)

mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = root_mean_squared_error(y_test, y_pred_base)
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print('=== БАЗОВАЯ МОДЕЛЬ (GRADIENT BOOSTING) ===')
print(f'MSE: {mse_base:.2f}')
print(f'RMSE: {rmse_base:.2f}')
print(f'MAE: {mae_base:.2f}')
print(f'R2: {r2_base:.4f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_base, color='royalblue', edgecolors='k', alpha=0.7)
min_val = min(y_test.min(), y_pred_base.min())
max_val = max(y_test.max(), y_pred_base.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('Реальная цена (Actual Price)')
plt.ylabel('Предсказанная цена (Predicted Price)')
plt.title('Базовая модель: Реальные vs Предсказанные цены')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_baseline_actual_vs_predicted.png'))
plt.close()

residuals_base = y_test - y_pred_base
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_base, residuals_base, color='crimson', edgecolors='k', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', lw=2)
plt.xlabel('Предсказанная цена (Predicted Price)')
plt.ylabel('Остатки (Residuals)')
plt.title('Базовая модель: График остатков')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_baseline_residuals.png'))
plt.close()

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

print('\n=== ПОДБОР ГИПЕРПАРАМЕТРОВ ЧЕРЕЗ GRIDSEARCHCV ===')
print(f'Лучшие параметры: {grid_search.best_params_}')
print(f'Лучший R2 на кросс-валидации: {grid_search.best_score_:.4f}')

y_pred_tuned = best_model.predict(X_test_scaled)

mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = root_mean_squared_error(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print('\n=== ОПТИМИЗИРОВАННАЯ МОДЕЛЬ (ПОСЛЕ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ) ===')
print(f'MSE: {mse_tuned:.2f}')
print(f'RMSE: {rmse_tuned:.2f}')
print(f'MAE: {mae_tuned:.2f}')
print(f'R2: {r2_tuned:.4f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_tuned, color='forestgreen', edgecolors='k', alpha=0.7)
min_val_t = min(y_test.min(), y_pred_tuned.min())
max_val_t = max(y_test.max(), y_pred_tuned.max())
plt.plot([min_val_t, max_val_t], [min_val_t, max_val_t], 'r--', lw=2)
plt.xlabel('Реальная цена (Actual Price)')
plt.ylabel('Предсказанная цена (Predicted Price)')
plt.title('Оптимизированная модель: Реальные vs Предсказанные цены')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_tuned_actual_vs_predicted.png'))
plt.close()

residuals_tuned = y_test - y_pred_tuned
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_tuned, residuals_tuned, color='purple', edgecolors='k', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', lw=2)
plt.xlabel('Предсказанная цена (Predicted Price)')
plt.ylabel('Остатки (Residuals)')
plt.title('Оптимизированная модель: График остатков')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_tuned_residuals.png'))
plt.close()

importances = best_model.feature_importances_
top_n = 15
indices = np.argsort(importances)[::-1][:top_n]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[indices][::-1], color='teal', align='center')
plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
plt.xlabel('Важность признака')
plt.title('Топ-15 наиболее важных признаков')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_feature_importance.png'))
plt.close()

learning_rates = [0.01, 0.05, 0.1, 0.2]
n_trees = 200

plt.figure(figsize=(10, 6))
for lr in learning_rates:
    gbr = GradientBoostingRegressor(
        n_estimators=n_trees,
        learning_rate=lr,
        max_depth=best_model.max_depth,
        min_samples_split=best_model.min_samples_split,
        min_samples_leaf=best_model.min_samples_leaf,
        random_state=42
    )
    gbr.fit(X_train_scaled, y_train)
    r2_scores = [r2_score(y_test, y_pred) for y_pred in gbr.staged_predict(X_test_scaled)]
    plt.plot(range(1, n_trees + 1), r2_scores, label=f'learning_rate = {lr}', lw=2)

plt.xlabel('Количество деревьев (n_estimators)')
plt.ylabel('Коэффициент детерминации R² на тестовой выборке')
plt.title('Влияние learning_rate и n_estimators на R²')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '06_learning_rate_trees_r2.png'))
plt.close()

plt.figure(figsize=(10, 6))
for lr in learning_rates:
    gbr = GradientBoostingRegressor(
        n_estimators=n_trees,
        learning_rate=lr,
        max_depth=best_model.max_depth,
        min_samples_split=best_model.min_samples_split,
        min_samples_leaf=best_model.min_samples_leaf,
        random_state=42
    )
    gbr.fit(X_train_scaled, y_train)
    rmse_scores = [root_mean_squared_error(y_test, y_pred) for y_pred in gbr.staged_predict(X_test_scaled)]
    plt.plot(range(1, n_trees + 1), rmse_scores, label=f'learning_rate = {lr}', lw=2)

plt.xlabel('Количество деревьев (n_estimators)')
plt.ylabel('RMSE на тестовой выборке')
plt.title('Влияние learning_rate и n_estimators на RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '07_learning_rate_trees_rmse.png'))
plt.close()
