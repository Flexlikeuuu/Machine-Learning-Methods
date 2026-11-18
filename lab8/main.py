import os
import sys

# Автоматическое переключение на lab8\.venv с поддержкой CUDA GPU
base_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(base_dir, '.venv', 'Scripts', 'python.exe')
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    import subprocess
    print(f"[*] Перенаправление запуска в виртуальное окружение GPU: {venv_python}")
    sys.exit(subprocess.call([venv_python] + sys.argv))

import shutil
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------------------
# 0. Настройка аппаратного ускорения (GPU CUDA / CPU) и директорий
# ------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_dir, 'plots')
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)
os.makedirs(plots_dir, exist_ok=True)

data_dir = os.path.join(base_dir, 'data')

print("=" * 75)
print(f"АППАРАТНОЕ УСКОРЕНИЕ: {device.type.upper()}")
if device.type == 'cuda':
    print(f"Видеокарта: {torch.cuda.get_device_name(0)}")
    print(f"Выделенная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB")
print("=" * 75)

# ------------------------------------------------------------------------------
# 1. Загрузка и подготовка набора данных CIFAR-10
# ------------------------------------------------------------------------------
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Аугментация и нормализация данных
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

print(f"Общее количество классов: {len(class_names)}")
print(f"Классы: {', '.join(class_names)}")
print(f"Обучающая выборка: {len(train_dataset)} изображений, тестовая: {len(test_dataset)} изображений")
print(f"Размер батча: {batch_size}")

# Визуализация примеров изображений 10 классов
raw_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
found_classes = set()
idx = 0
while len(found_classes) < 10 and idx < len(raw_cifar):
    img, lbl = raw_cifar[idx]
    if lbl not in found_classes:
        ax = axes[lbl]
        ax.imshow(img)
        ax.set_title(f"{lbl}: {class_names[lbl]}", fontsize=11, fontweight='bold')
        ax.axis('off')
        found_classes.add(lbl)
    idx += 1
plt.suptitle('Примеры изображений 10 классов датасета CIFAR-10 (32x32 RGB)', fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_cifar10_sample_classes.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 2. Определение архитектур сверточных нейронных сетей (CNN)
# ------------------------------------------------------------------------------

# Базовая архитектура (Baseline CNN): Conv -> Pool -> Conv -> Pool -> FC -> Dropout -> FC
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(self.pool2(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Улучшенная архитектура (Improved CNN): Conv-BN-ReLU x2 -> Pool (3 блока: 32->64->128) + Dense-BN-Dropout
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Блок 1: 32 фильтра
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.2)

        # Блок 2: 64 фильтра
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.3)

        # Блок 3: 128 фильтров
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.4)

        # Классификатор
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(F.relu(self.bn2(self.conv2(x)))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop2(self.pool2(F.relu(self.bn4(self.conv4(x)))))

        x = self.drop3(self.pool3(F.relu(self.bn5(self.conv5(x)))))

        x = x.view(x.size(0), -1)
        x = self.drop_fc(F.relu(self.bn_fc(self.fc1(x))))
        x = self.fc2(x)
        return x

def train_and_eval(model, train_loader, test_loader, epochs=15, lr=0.001, weight_decay=0.0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Валидация на тестовой выборке
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        ep_time = time.time() - t_start
        print(f"Эпоха [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}с) | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

    return history

# ------------------------------------------------------------------------------
# 3. Обучение моделей на GPU
# ------------------------------------------------------------------------------
epochs_cnt = 15

print("\n" + "=" * 75)
print(f"2. ОБУЧЕНИЕ БАЗОВОЙ МОДЕЛИ (BASELINE CNN НА {device.type.upper()})")
print("=" * 75)
baseline_model = BaselineCNN()
t0 = time.time()
base_hist = train_and_eval(baseline_model, train_loader, test_loader, epochs=epochs_cnt, lr=0.001)
print(f"Общее время обучения базовой модели: {time.time() - t0:.1f} сек")

print("\n" + "=" * 75)
print(f"3. ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ (IMPROVED CNN НА {device.type.upper()})")
print("=" * 75)
improved_model = ImprovedCNN()
t0 = time.time()
improved_hist = train_and_eval(improved_model, train_loader, test_loader, epochs=epochs_cnt, lr=0.001, weight_decay=1e-4)
print(f"Общее время обучения улучшенной модели: {time.time() - t0:.1f} сек")

# ------------------------------------------------------------------------------
# 4. Графики динамики обучения
# ------------------------------------------------------------------------------
epochs_axis = range(1, epochs_cnt + 1)

# График функции потерь
plt.figure(figsize=(10, 5))
plt.plot(epochs_axis, base_hist['train_loss'], 'b--', label='Baseline: Train Loss')
plt.plot(epochs_axis, base_hist['val_loss'], 'b-', lw=2, label='Baseline: Val Loss')
plt.plot(epochs_axis, improved_hist['train_loss'], 'g--', label='Improved: Train Loss')
plt.plot(epochs_axis, improved_hist['val_loss'], 'g-', lw=2, label='Improved: Val Loss')
plt.title('Динамика функции потерь (Cross-Entropy Loss)', fontsize=12)
plt.xlabel('Эпоха обучения')
plt.ylabel('Loss')
plt.xticks(epochs_axis)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_training_dynamics_loss.png'), dpi=300)
plt.close()

# График точности
plt.figure(figsize=(10, 5))
plt.plot(epochs_axis, [x * 100 for x in base_hist['train_acc']], 'b--', label='Baseline: Train Acc')
plt.plot(epochs_axis, [x * 100 for x in base_hist['val_acc']], 'b-', lw=2, label='Baseline: Val Acc')
plt.plot(epochs_axis, [x * 100 for x in improved_hist['train_acc']], 'g--', label='Improved: Train Acc')
plt.plot(epochs_axis, [x * 100 for x in improved_hist['val_acc']], 'g-', lw=2, label='Improved: Val Acc')
plt.title('Динамика точности классификации (Accuracy, %)', fontsize=12)
plt.xlabel('Эпоха обучения')
plt.ylabel('Точность (%)')
plt.xticks(epochs_axis)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_training_dynamics_accuracy.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 5. Оценка качества: Матрица ошибок и примеры предсказаний
# ------------------------------------------------------------------------------
improved_model.eval()
all_preds = []
all_targets = []
sample_imgs = []
sample_preds = []
sample_true = []

with torch.no_grad():
    for images, labels in test_loader:
        images_dev = images.to(device)
        outputs = improved_model(images_dev)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())

        if len(sample_imgs) < 16:
            for i in range(images.size(0)):
                if len(sample_imgs) < 16:
                    sample_imgs.append(images[i])
                    sample_preds.append(preds[i].item())
                    sample_true.append(labels[i].item())

cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок (Confusion Matrix) улучшенной модели CNN', fontsize=12)
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=9)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_confusion_matrix.png'), dpi=300)
plt.close()

# Примеры предсказаний
fig, axes = plt.subplots(4, 4, figsize=(11, 11))
axes = axes.flatten()
inv_mean = np.array([0.4914, 0.4822, 0.4465])
inv_std = np.array([0.2470, 0.2435, 0.2616])

for i in range(16):
    img_np = sample_imgs[i].cpu().numpy().transpose(1, 2, 0)
    img_denorm = np.clip(img_np * inv_std + inv_mean, 0, 1)
    ax = axes[i]
    ax.imshow(img_denorm)
    is_correct = (sample_preds[i] == sample_true[i])
    color = 'green' if is_correct else 'red'
    ax.set_title(f"P: {class_names[sample_preds[i]]}\nT: {class_names[sample_true[i]]}",
                 color=color, fontsize=9, fontweight='bold')
    ax.axis('off')
plt.suptitle('Примеры предсказаний тестовых изображений (зеленый – верно, красный – ошибка)', fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_sample_predictions.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 6. Визуализация фильтров и карт признаков
# ------------------------------------------------------------------------------
filters = improved_model.conv1.weight.data.cpu()
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
axes = axes.flatten()
for i in range(32):
    f_img = filters[i].numpy().transpose(1, 2, 0)
    f_norm = (f_img - f_img.min()) / (f_img.max() - f_img.min() + 1e-8)
    axes[i].imshow(f_norm)
    axes[i].axis('off')
    axes[i].set_title(f"Ф.{i+1}", fontsize=8)
plt.suptitle('Визуализация 32 фильтров первого сверточного слоя (3x3 ядра)', fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '06_first_layer_filters.png'), dpi=300)
plt.close()

test_img = sample_imgs[0].unsqueeze(0).to(device)
with torch.no_grad():
    feat_map = F.relu(improved_model.bn1(improved_model.conv1(test_img))).squeeze(0).cpu()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
axes = axes.flatten()
for i in range(32):
    fm = feat_map[i].numpy()
    axes[i].imshow(fm, cmap='viridis')
    axes[i].axis('off')
    axes[i].set_title(f"К.{i+1}", fontsize=8)
plt.suptitle('Карты признаков (Feature Maps) первого слоя после активации ReLU', fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '07_feature_maps.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 7. ЧАСТЬ 2: Реализация оптимизаторов (Adam, AdamW, AMSGrad) с нуля
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("4. ЧАСТЬ 2: СРАВНЕНИЕ ОПТИМИЗАТОРОВ (ADAM, ADAMW, AMSGRAD)")
print("=" * 75)

def rosenbrock(theta):
    x, y = theta[0], theta[1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def rosenbrock_grad(theta):
    x, y = theta[0], theta[1]
    gx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    gy = 200.0 * (y - x**2)
    return np.array([gx, gy], dtype=np.float64)

def run_optimizer(opt_type, theta_init, iterations=50, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.02):
    theta = theta_init.copy().astype(np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    v_max = np.zeros_like(theta)

    history = {
        'thetas': [theta.copy()],
        'losses': [rosenbrock(theta)],
        'norms': [np.linalg.norm(theta)],
        'step_sizes': []
    }

    for t in range(1, iterations + 1):
        g = rosenbrock_grad(theta)
        prev_theta = theta.copy()

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        if opt_type == 'Adam':
            g_reg = g + weight_decay * theta
            m_reg = beta1 * m + (1.0 - beta1) * g_reg
            m_hat_reg = m_reg / (1.0 - beta1**t)
            theta = theta - lr * m_hat_reg / (np.sqrt(v_hat) + eps)

        elif opt_type == 'AdamW':
            theta = theta - lr * weight_decay * theta - lr * m_hat / (np.sqrt(v_hat) + eps)

        elif opt_type == 'AMSGrad':
            v_max = np.maximum(v_max, v)
            theta = theta - lr * m_hat / (np.sqrt(v_max) + eps)

        step_sz = np.linalg.norm(theta - prev_theta)
        history['thetas'].append(theta.copy())
        history['losses'].append(rosenbrock(theta))
        history['norms'].append(np.linalg.norm(theta))
        history['step_sizes'].append(step_sz)

    return history

theta_start = np.array([-1.2, 1.5], dtype=np.float64)
n_iters = 50

res_adam = run_optimizer('Adam', theta_start, iterations=n_iters, lr=0.005, weight_decay=0.02)
res_adamw = run_optimizer('AdamW', theta_start, iterations=n_iters, lr=0.005, weight_decay=0.02)
res_amsgrad = run_optimizer('AMSGrad', theta_start, iterations=n_iters, lr=0.005)

opt_results = [
    {'Оптимизатор': 'Adam', 'Начальный Loss': res_adam['losses'][0], 'Loss на 50 итер.': res_adam['losses'][-1], 'Финальная норма весов': res_adam['norms'][-1]},
    {'Оптимизатор': 'AdamW', 'Начальный Loss': res_adamw['losses'][0], 'Loss на 50 итер.': res_adamw['losses'][-1], 'Финальная норма весов': res_adamw['norms'][-1]},
    {'Оптимизатор': 'AMSGrad', 'Начальный Loss': res_amsgrad['losses'][0], 'Loss на 50 итер.': res_amsgrad['losses'][-1], 'Финальная норма весов': res_amsgrad['norms'][-1]}
]
opt_df = pd.DataFrame(opt_results)
print(opt_df.to_string(index=False))

# Траектории в пространстве параметров
x_grid = np.linspace(-1.8, 1.8, 400)
y_grid = np.linspace(-0.5, 2.5, 400)
X_m, Y_m = np.meshgrid(x_grid, y_grid)
Z_m = (1.0 - X_m)**2 + 100.0 * (Y_m - X_m**2)**2

plt.figure(figsize=(9, 7))
plt.contour(X_m, Y_m, Z_m, levels=np.logspace(0, 4, 35), cmap='Spectral_r', alpha=0.6)

th_adam = np.array(res_adam['thetas'])
th_adamw = np.array(res_adamw['thetas'])
th_amsgrad = np.array(res_amsgrad['thetas'])

plt.plot(th_adam[:, 0], th_adam[:, 1], 'ro-', label='Adam', lw=2, markersize=3)
plt.plot(th_adamw[:, 0], th_adamw[:, 1], 'bs-', label='AdamW', lw=2, markersize=3)
plt.plot(th_amsgrad[:, 0], th_amsgrad[:, 1], 'g^-', label='AMSGrad', lw=2, markersize=3)

plt.plot(theta_start[0], theta_start[1], 'k*', markersize=14, label='Старт (-1.2, 1.5)')
plt.plot(1.0, 1.0, 'kX', markersize=12, label='Глобальный оптимум (1, 1)')
plt.title('Траектории изменения весов оптимизаторов за 50 итераций', fontsize=12)
plt.xlabel('Параметр $\\theta_1$')
plt.ylabel('Параметр $\\theta_2$')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '08_optimizers_trajectories_2d.png'), dpi=300)
plt.close()

# График сходимости потерь
iter_axis = range(0, n_iters + 1)
plt.figure(figsize=(9, 5))
plt.plot(iter_axis, res_adam['losses'], 'r-', lw=2, label='Adam')
plt.plot(iter_axis, res_adamw['losses'], 'b-', lw=2, label='AdamW')
plt.plot(iter_axis, res_amsgrad['losses'], 'g-', lw=2, label='AMSGrad')
plt.yscale('log')
plt.title('Сходимость функции потерь Loss на 50 итерациях (лог-масштаб)', fontsize=12)
plt.xlabel('Итерация')
plt.ylabel('Значение Loss (Log)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '09_optimizers_loss_convergence.png'), dpi=300)
plt.close()

# График нормы весов
plt.figure(figsize=(9, 5))
plt.plot(iter_axis, res_adam['norms'], 'r-', lw=2, label='Adam')
plt.plot(iter_axis, res_adamw['norms'], 'b-', lw=2, label='AdamW')
plt.plot(iter_axis, res_amsgrad['norms'], 'g-', lw=2, label='AMSGrad')
plt.title(r'Динамика евклидовой нормы вектора весов $||\mathbf{\theta}_t||_2$', fontsize=12)
plt.xlabel('Итерация')
plt.ylabel('$||\\mathbf{\\theta}_t||_2$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '10_optimizers_weights_norm.png'), dpi=300)
plt.close()

# График шага обновления весов
plt.figure(figsize=(9, 5))
plt.plot(range(1, n_iters + 1), res_adam['step_sizes'], 'r-', lw=1.5, label='Adam')
plt.plot(range(1, n_iters + 1), res_adamw['step_sizes'], 'b-', lw=1.5, label='AdamW')
plt.plot(range(1, n_iters + 1), res_amsgrad['step_sizes'], 'g-', lw=1.5, label='AMSGrad')
plt.title(r'Величина шага обновления весов $||\mathbf{\theta}_t - \mathbf{\theta}_{t-1}||_2$', fontsize=12)
plt.xlabel('Итерация')
plt.ylabel('Шаг обновления')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '11_optimizers_step_size.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 8. Сводная таблица результатов CNN
# ------------------------------------------------------------------------------
cnn_summary = [
    {
        'Модель': 'Baseline CNN (2 Conv, Dropout)',
        'Финальный Train Loss': round(base_hist['train_loss'][-1], 4),
        'Финальный Val Loss': round(base_hist['val_loss'][-1], 4),
        'Train Accuracy (%)': round(base_hist['train_acc'][-1] * 100, 2),
        'Test Accuracy (%)': round(base_hist['val_acc'][-1] * 100, 2)
    },
    {
        'Модель': 'Improved CNN (BatchNorm, 3 блока, L2)',
        'Финальный Train Loss': round(improved_hist['train_loss'][-1], 4),
        'Финальный Val Loss': round(improved_hist['val_loss'][-1], 4),
        'Train Accuracy (%)': round(improved_hist['train_acc'][-1] * 100, 2),
        'Test Accuracy (%)': round(improved_hist['val_acc'][-1] * 100, 2)
    }
]
cnn_df = pd.DataFrame(cnn_summary)
print("\n" + "=" * 75)
print("5. СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ CNN НА CIFAR-10")
print("=" * 75)
print(cnn_df.to_string(index=False))

print(f"\nВсе 11 графиков успешно сохранены в директорию: {plots_dir}")
