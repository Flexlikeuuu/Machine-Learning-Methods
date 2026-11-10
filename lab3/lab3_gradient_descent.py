import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

csv_path = "CarPrice_Assignment.csv"
df = pd.read_csv(csv_path)

feature_cols = ['enginesize', 'curbweight', 'horsepower']
target_col = 'price'

X_raw = df[feature_cols].values
y_raw = df[target_col].values

X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X_scaled = (X_raw - X_mean) / X_std

y_mean = np.mean(y_raw)
y_std = np.std(y_raw)
y_scaled = (y_raw - y_mean) / y_std

n_samples = X_scaled.shape[0]
X_design = np.c_[np.ones(n_samples), X_scaled]
feature_names = ['bias (w0)', 'enginesize (w1)', 'curbweight (w2)', 'horsepower (w3)']

w_ols = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_scaled
mse_ols = np.mean((X_design @ w_ols - y_scaled) ** 2)

out_text_1 = (
    f"--- 1. ИМПОРТ И ПОДГОТОВКА ДАННЫХ ---\n"
    f"Датасет: {csv_path}\n"
    f"Размерность: {df.shape[0]} строк, {df.shape[1]} столбцов\n"
    f"Выбранные признаки: {feature_cols}\n"
    f"Целевой признак: {target_col}\n\n"
    f"Размерность матрицы плана X: {X_design.shape} (включая фиктивный признак x0 = 1)\n"
    f"Первые 3 строки стандартизированных признаков X (без x0):\n"
    f"{pd.DataFrame(X_scaled[:3], columns=feature_cols).to_string(index=True)}\n\n"
    f"Аналитическое решение МНК (w = (X^T * X)^(-1) * X^T * y):\n"
    f"  w0 (смещение):          {w_ols[0]:10.5f}\n"
    f"  w1 (enginesize):        {w_ols[1]:10.5f}\n"
    f"  w2 (curbweight):        {w_ols[2]:10.5f}\n"
    f"  w3 (horsepower):        {w_ols[3]:10.5f}\n"
    f"  Минимальное MSE (МНК):  {mse_ols:10.5f}"
)
print(out_text_1)

def compute_mse(X, y, w):
    return np.mean((X @ w - y) ** 2)

def gradient_descent(X, y, lr=0.1, max_iter=1000, eps=1e-5):
    n, m = X.shape
    w = np.zeros(m)
    mse_history = [compute_mse(X, y, w)]
    w_history = [w.copy()]
    
    for k in range(1, max_iter + 1):
        gradient = (2.0 / n) * (X.T @ (X @ w - y))
        w_new = w - lr * gradient
        diff = np.linalg.norm(w_new - w)
        
        w = w_new
        w_history.append(w.copy())
        mse_history.append(compute_mse(X, y, w))
        
        if diff < eps:
            return w, k, mse_history, w_history
            
    return w, max_iter, mse_history, w_history

w_gd, iters_gd, mse_hist_gd, _ = gradient_descent(X_design, y_scaled, lr=0.1, eps=1e-5)

out_text_2 = (
    f"--- 2. КЛАССИЧЕСКИЙ ГРАДИЕНТНЫЙ СПУСК ---\n"
    f"Параметры: learning_rate = 0.1, eps = 1e-05, max_iter = 1000\n"
    f"Сходимость достигнута за: {iters_gd} итераций\n\n"
    f"Полученные веса:\n"
    f"  w0 (смещение):   {w_gd[0]:10.5f}  (МНК: {w_ols[0]:10.5f})\n"
    f"  w1 (enginesize): {w_gd[1]:10.5f}  (МНК: {w_ols[1]:10.5f})\n"
    f"  w2 (curbweight): {w_gd[2]:10.5f}  (МНК: {w_ols[2]:10.5f})\n"
    f"  w3 (horsepower): {w_gd[3]:10.5f}  (МНК: {w_ols[3]:10.5f})\n\n"
    f"Итоговая ошибка MSE: {mse_hist_gd[-1]:.6f} (МНК: {mse_ols:.6f})\n"
    f"Максимальная разность с МНК: {np.max(np.abs(w_gd - w_ols)):.2e}"
)
print("\n" + out_text_2)

Hessian = (2.0 / n_samples) * (X_design.T @ X_design)
max_eigenval = np.max(np.linalg.eigvalsh(Hessian))
lr_crit = 2.0 / max_eigenval

learning_rates = [0.005, 0.02, 0.08, 0.20, 0.35, 0.40]
lr_results = {}

for lr in learning_rates:
    w_res, iters_res, mse_hist_res, _ = gradient_descent(X_design, y_scaled, lr=lr, max_iter=250, eps=1e-5)
    lr_results[lr] = (iters_res, mse_hist_res, w_res)

plt.figure(figsize=(9.2, 5.4), dpi=300)
colors_lr = ['#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#8c564b']

for lr, col in zip(learning_rates, colors_lr):
    iters_res, mse_hist, _ = lr_results[lr]
    limit_k = min(120, len(mse_hist))
    if np.isnan(mse_hist[-1]) or mse_hist[-1] > 5.0:
        label_text = f'lr = {lr} (расходимость)'
    elif iters_res >= 250:
        label_text = f'lr = {lr} (медленная сходимость, >250 итер.)'
    else:
        label_text = f'lr = {lr} (сходимость за {iters_res} итер.)'
    plt.plot(range(limit_k), mse_hist[:limit_k], lw=2.0, color=col, label=label_text)

plt.axhline(mse_ols, color='black', linestyle=':', lw=1.2, label=f'МНК минимум: {mse_ols:.4f}')
plt.title("Влияние скорости обучения (learning rate) на сходимость градиентного спуска", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Номер итерации", fontsize=11)
plt.ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
plt.ylim(0.1, 1.1)
plt.xlim(0, 110)
plt.grid(True)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
plt.tight_layout()
plt.savefig("fig_learning_rates.png", dpi=300)
plt.close()

def stochastic_gradient_descent(X, y, lr=0.015, max_iter=1200, seed=42):
    np.random.seed(seed)
    n, m = X.shape
    w = np.zeros(m)
    mse_history = [compute_mse(X, y, w)]
    w_history = [w.copy()]
    
    for k in range(1, max_iter + 1):
        i = np.random.randint(0, n)
        xi = X[i]
        yi = y[i]
        
        pred_i = np.dot(xi, w)
        grad_i = 2.0 * xi * (pred_i - yi)
        
        lr_k = lr / (1.0 + 0.002 * k)
        w = w - lr_k * grad_i
        
        w_history.append(w.copy())
        mse_history.append(compute_mse(X, y, w))
            
    return w, max_iter, mse_history, w_history

w_sgd, iters_sgd, mse_hist_sgd, _ = stochastic_gradient_descent(X_design, y_scaled, lr=0.015, max_iter=1200)

out_text_3 = (
    f"--- 3. СТОХАСТИЧЕСКИЙ ГРАДИЕНТНЫЙ СПУСК (SGD) ---\n"
    f"Параметры: lr0 = 0.015, затухание lr_k = lr0 / (1 + 0.002*k), итераций = 1200\n"
    f"Итоговые веса SGD:\n"
    f"  w0 (смещение):   {w_sgd[0]:10.5f}  (МНК: {w_ols[0]:10.5f})\n"
    f"  w1 (enginesize): {w_sgd[1]:10.5f}  (МНК: {w_ols[1]:10.5f})\n"
    f"  w2 (curbweight): {w_sgd[2]:10.5f}  (МНК: {w_ols[2]:10.5f})\n"
    f"  w3 (horsepower): {w_sgd[3]:10.5f}  (МНК: {w_ols[3]:10.5f})\n\n"
    f"Итоговая ошибка MSE (SGD): {mse_hist_sgd[-1]:.6f} (МНК: {mse_ols:.6f})\n"
    f"Разница MSE между SGD и оптимумом МНК: {mse_hist_sgd[-1] - mse_ols:.5f}"
)
print("\n" + out_text_3)

plt.figure(figsize=(9, 5.2), dpi=300)
n_cmp = 350

plt.plot(range(min(n_cmp, len(mse_hist_gd))), mse_hist_gd[:n_cmp], 
         label='Пакетный градиентный спуск (GD)', color='#1f77b4', lw=2.2)
plt.plot(range(n_cmp), mse_hist_sgd[:n_cmp], 
         label='Стохастический градиентный спуск (SGD)', color='#ff7f0e', lw=1.2, alpha=0.85)

plt.axhline(y=mse_ols, color='red', linestyle=':', lw=1.5, label=f'Аналитический оптимум МНК ({mse_ols:.4f})')

plt.title("Сравнение динамики MSE: Пакетный градиентный спуск vs SGD", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Номер итерации (шаг обновления весов)", fontsize=11)
plt.ylabel("Среднеквадратичная ошибка (MSE)", fontsize=11)
plt.ylim(0.12, 1.05)
plt.xlim(0, n_cmp)
plt.grid(True)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig("fig_gd_vs_sgd.png", dpi=300)
plt.close()

lambdas = np.logspace(-3, 3, 20)

def gradient_descent_l2(X, y, lmb, max_iter=3000, eps=1e-6):
    n, m = X.shape
    w = np.zeros(m)
    lr = min(0.08, 0.4 / (1.0 + 2.0 * lmb))
    
    for it in range(max_iter):
        w_reg = w.copy()
        w_reg[0] = 0.0
        grad = (2.0 / n) * (X.T @ (X @ w - y)) + 2.0 * lmb * w_reg
        w_new = w - lr * grad
        if np.linalg.norm(w_new - w) < eps:
            return w_new
        w = w_new
    return w

I_reg = np.eye(X_design.shape[1])
I_reg[0, 0] = 0.0

weights_l2 = []
analytical_l2 = []

for lmb in lambdas:
    w_ana = np.linalg.inv(X_design.T @ X_design + n_samples * lmb * I_reg) @ X_design.T @ y_scaled
    analytical_l2.append(w_ana)
    w_num = gradient_descent_l2(X_design, y_scaled, lmb)
    weights_l2.append(w_num)

weights_l2 = np.array(weights_l2)
analytical_l2 = np.array(analytical_l2)

out_text_4 = (
    f"--- 4. L2-РЕГУЛЯРИЗАЦИЯ (RIDGE) ---\n"
    f"Диапазон 20 значений lambda: от {lambdas[0]:.3f} до {lambdas[-1]:.1f}\n"
    f"Максимальное расхождение с точным аналитическим Ridge: {np.max(np.abs(weights_l2 - analytical_l2)):.2e}\n\n"
    f"Выборка значений весов при различных коэффициентах lambda:\n"
    f"{'lambda':>10s} | {'w0 (bias)':>10s} | {'w1 (engine)':>12s} | {'w2 (weight)':>12s} | {'w3 (power)':>12s}\n"
    f"{'-'*65}\n"
)
indices_sample = [0, 4, 8, 12, 16, 19]
for idx in indices_sample:
    lmb_val = lambdas[idx]
    w_row = weights_l2[idx]
    out_text_4 += f"{lmb_val:10.4f} | {w_row[0]:10.5f} | {w_row[1]:12.5f} | {w_row[2]:12.5f} | {w_row[3]:12.5f}\n"

print("\n" + out_text_4)

plt.figure(figsize=(9.2, 5.6), dpi=300)
colors_w = ['#7f7f7f', '#1f77b4', '#2ca02c', '#d62728']
markers_w = ['o', 's', '^', 'v']

for j in range(X_design.shape[1]):
    plt.semilogx(lambdas, weights_l2[:, j], 
                 label=feature_names[j], 
                 color=colors_w[j], 
                 marker=markers_w[j], 
                 markersize=5, 
                 lw=1.8)

plt.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.6)
plt.title(r"Траектории весов модели в зависимости от коэффициента L2-регуляризации ($\lambda$)", 
          fontsize=12, fontweight='bold', pad=12)
plt.xlabel(r"Коэффициент регуляризации $\lambda$ (логарифмическая шкала)", fontsize=11)
plt.ylabel(r"Значение весового коэффициента $w_j$", fontsize=11)
plt.grid(True, which="both", ls="--")
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig("fig_l2_regularization.png", dpi=300)
plt.close()
