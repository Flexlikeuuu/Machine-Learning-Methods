import os
import sys

# Автоматический перезапуск в изолированном виртуальном окружении lab9\.venv с поддержкой CUDA
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
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------------------
# 0. Настройка аппаратного ускорения (GPU CUDA / CPU) и директорий
# ------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# 1. Загрузка и подготовка набора данных Fashion-MNIST
# ------------------------------------------------------------------------------
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Нормализация пикселей к диапазону [-1, 1] для согласования с активацией tanh генератора
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

print(f"Набор данных: Fashion-MNIST (60000 изображений 28x28 в градациях серого)")
print(f"Количество классов: {len(class_names)} ({', '.join(class_names)})")
print(f"Размер батча: {batch_size}, общее число батчей в эпохе: {len(dataloader)}")

# Визуализация образцов реальных данных
raw_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
found_classes = set()
idx = 0
while len(found_classes) < 10 and idx < len(raw_dataset):
    img, lbl = raw_dataset[idx]
    if lbl not in found_classes:
        ax = axes[lbl]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{lbl}: {class_names[lbl]}", fontsize=11, fontweight='bold')
        ax.axis('off')
        found_classes.add(lbl)
    idx += 1
plt.suptitle('Образцы реальных изображений 10 классов Fashion-MNIST (28x28)', fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_real_fashion_mnist_samples.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 2. Архитектура генератора и дискриминатора DCGAN
# ------------------------------------------------------------------------------
latent_dim = 100

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# Генератор: преобразует вектор шума z (100) в изображение 1x28x28
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.init_size = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Дискриминатор: классифицирует изображение 1x28x28 как реальное или сгенерированное
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# ------------------------------------------------------------------------------
# 3. Обучение моделей: Классический DCGAN (BCE) и LSGAN (MSE)
# ------------------------------------------------------------------------------
fixed_noise = torch.randn(64, latent_dim, device=device)

def train_gan(loss_type='bce', epochs=15, lr=0.0002):
    print("\n" + "=" * 75)
    print(f"ОБУЧЕНИЕ МОДЕЛИ: {loss_type.upper()} GAN НА {device.type.upper()} ({epochs} эпох)")
    print("=" * 75)
    
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    if loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
        
    history = {'epoch': [], 'loss_D': [], 'loss_G': [], 'd_real_acc': [], 'd_fake_acc': [], 'samples': {}}
    
    t_start_total = time.time()
    
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        d_losses = []
        g_losses = []
        real_correct = 0
        fake_correct = 0
        total_samples = 0
        
        for i, (imgs, _) in enumerate(dataloader):
            batch_size_cur = imgs.size(0)
            real_imgs = imgs.to(device)
            
            # Разметка классов
            if loss_type == 'bce':
                # Одностороннее сглаживание меток (0.9 для реальных)
                valid = torch.full((batch_size_cur, 1), 0.9, device=device)
                fake = torch.full((batch_size_cur, 1), 0.0, device=device)
            else:
                valid = torch.full((batch_size_cur, 1), 1.0, device=device)
                fake = torch.full((batch_size_cur, 1), 0.0, device=device)
                
            # ---------------------
            #  Обучение Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Генерация батча фейковых изображений
            z = torch.randn(batch_size_cur, latent_dim, device=device)
            gen_imgs = generator(z)
            
            real_preds = discriminator(real_imgs)
            fake_preds = discriminator(gen_imgs.detach())
            
            loss_real = criterion(real_preds, valid)
            loss_fake = criterion(fake_preds, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            optimizer_D.step()
            
            # Метрики точности дискриминатора
            if loss_type == 'bce':
                real_acc = (torch.sigmoid(real_preds) > 0.5).float().sum().item()
                fake_acc = (torch.sigmoid(fake_preds) < 0.5).float().sum().item()
            else:
                real_acc = (real_preds > 0.5).float().sum().item()
                fake_acc = (fake_preds < 0.5).float().sum().item()
            real_correct += real_acc
            fake_correct += fake_acc
            total_samples += batch_size_cur
            
            # -----------------
            #  Обучение Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Для генератора метка - целевой класс 1.0 (valid)
            gen_preds = discriminator(gen_imgs)
            if loss_type == 'bce':
                loss_G = criterion(gen_preds, torch.ones_like(gen_preds))
            else:
                loss_G = criterion(gen_preds, valid)
                
            loss_G.backward()
            optimizer_G.step()
            
            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())
            
        epoch_d_loss = np.mean(d_losses)
        epoch_g_loss = np.mean(g_losses)
        epoch_d_real_acc = (real_correct / total_samples) * 100.0
        epoch_d_fake_acc = (fake_correct / total_samples) * 100.0
        
        history['epoch'].append(epoch)
        history['loss_D'].append(epoch_d_loss)
        history['loss_G'].append(epoch_g_loss)
        history['d_real_acc'].append(epoch_d_real_acc)
        history['d_fake_acc'].append(epoch_d_fake_acc)
        
        ep_time = time.time() - t_epoch
        print(f"Эпоха [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}с) | Loss D: {epoch_d_loss:.4f} (Real Acc: {epoch_d_real_acc:.1f}%, Fake Acc: {epoch_d_fake_acc:.1f}%) | Loss G: {epoch_g_loss:.4f}")
        
        # Сохранение сгенерированных изображений на ключевых эпохах
        if epoch in [1, 3, 5, 10]:
            generator.eval()
            with torch.no_grad():
                gen_samples = generator(fixed_noise).cpu()
                history['samples'][epoch] = gen_samples
            generator.train()
            
    print(f"Общее время обучения {loss_type.upper()}: {time.time() - t_start_total:.1f} сек")
    return generator, discriminator, history

# Запуск обучения DCGAN и LSGAN
epochs_cnt = 10
dcgan_G, dcgan_D, dcgan_hist = train_gan(loss_type='bce', epochs=epochs_cnt, lr=0.0002)
lsgan_G, lsgan_D, lsgan_hist = train_gan(loss_type='lsgan', epochs=epochs_cnt, lr=0.0002)

# ------------------------------------------------------------------------------
# 4. Построение графиков и визуализация результатов
# ------------------------------------------------------------------------------
epochs_axis = range(1, epochs_cnt + 1)

# График 02: Потери DCGAN
plt.figure(figsize=(9, 5))
plt.plot(epochs_axis, dcgan_hist['loss_D'], 'r-o', lw=2, label='Loss Discriminator (BCE)')
plt.plot(epochs_axis, dcgan_hist['loss_G'], 'b-s', lw=2, label='Loss Generator (BCE)')
plt.title('Динамика функций потерь DCGAN (Loss D и Loss G)', fontsize=12)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.xticks(epochs_axis)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_dcgan_training_losses.png'), dpi=300)
plt.close()

# График 03: Потери LSGAN
plt.figure(figsize=(9, 5))
plt.plot(epochs_axis, lsgan_hist['loss_D'], 'crimson', marker='o', lw=2, label='Loss Discriminator (MSE)')
plt.plot(epochs_axis, lsgan_hist['loss_G'], 'royalblue', marker='s', lw=2, label='Loss Generator (MSE)')
plt.title('Динамика функций потерь LSGAN (Least Squares GAN, MSE Loss)', fontsize=12)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.xticks(epochs_axis)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_lsgan_training_losses.png'), dpi=300)
plt.close()

# График 04: Сравнение динамики потерь DCGAN vs LSGAN
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs_axis, dcgan_hist['loss_D'], 'r-', lw=2, label='DCGAN Loss D')
ax1.plot(epochs_axis, lsgan_hist['loss_D'], 'g-', lw=2, label='LSGAN Loss D')
ax1.set_title('Сравнение потерь Дискриминатора (D Loss)', fontsize=11)
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Loss')
ax1.set_xticks(epochs_axis)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_axis, dcgan_hist['loss_G'], 'b-', lw=2, label='DCGAN Loss G')
ax2.plot(epochs_axis, lsgan_hist['loss_G'], 'm-', lw=2, label='LSGAN Loss G')
ax2.set_title('Сравнение потерь Генератора (G Loss)', fontsize=11)
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Loss')
ax2.set_xticks(epochs_axis)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.suptitle('Сравнительная сходимость функций потерь: DCGAN vs LSGAN', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_dcgan_loss_comparison.png'), dpi=300)
plt.close()

# Вспомогательная функция для сохранения сетки 8x8 сгенерированных изображений
def save_image_grid(tensor_imgs, save_path, title):
    # Денормализация из [-1, 1] в [0, 1]
    imgs = (tensor_imgs[:64] + 1.0) / 2.0
    grid = make_grid(imgs, nrow=8, padding=2, normalize=False)
    np_grid = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid[:, :, 0], cmap='gray')
    plt.title(title, fontsize=12, pad=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Графики 05, 06, 07: Эволюция генерации DCGAN
save_image_grid(dcgan_hist['samples'][1], os.path.join(plots_dir, '05_dcgan_generated_epoch_01.png'),
                'Сгенерированные изображения DCGAN (Эпоха 1 — шум и начальные контуры)')
save_image_grid(dcgan_hist['samples'][5], os.path.join(plots_dir, '06_dcgan_generated_epoch_05.png'),
                'Сгенерированные изображения DCGAN (Эпоха 5 — формирование силуэтов одежды)')
save_image_grid(dcgan_hist['samples'][10], os.path.join(plots_dir, '07_dcgan_generated_epoch_10.png'),
                'Сгенерированные изображения DCGAN (Эпоха 10 — четкие предметы гардероба)')

# График 08: Финальная генерация LSGAN
save_image_grid(lsgan_hist['samples'][10], os.path.join(plots_dir, '08_lsgan_generated_epoch_10.png'),
                'Сгенерированные изображения LSGAN (Эпоха 10 — высокая резкость и контраст)')

# График 09: Сравнительная сетка «Реальные vs DCGAN vs LSGAN»
real_batch = next(iter(dataloader))[0][:16]
real_grid_data = (real_batch + 1.0) / 2.0
dcgan_16 = (dcgan_hist['samples'][10][:16] + 1.0) / 2.0
lsgan_16 = (lsgan_hist['samples'][10][:16] + 1.0) / 2.0

fig, axes = plt.subplots(3, 8, figsize=(14, 6))
for i in range(8):
    # Строка 1: Реальные
    axes[0, i].imshow(real_grid_data[i, 0].numpy(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Реальные (Fashion-MNIST)', fontsize=10, loc='left', fontweight='bold')
        
    # Строка 2: DCGAN
    axes[1, i].imshow(dcgan_16[i, 0].numpy(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Сгенерированные DCGAN', fontsize=10, loc='left', fontweight='bold')
        
    # Строка 3: LSGAN
    axes[2, i].imshow(lsgan_16[i, 0].numpy(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Сгенерированные LSGAN', fontsize=10, loc='left', fontweight='bold')

plt.suptitle('Сравнительное сопоставление: Реальные изображения vs DCGAN vs LSGAN', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '09_comparison_real_vs_dcgan_vs_lsgan.png'), dpi=300)
plt.close()

# График 10: Гистограмма попиксельного распределения яркости
real_pixels = ((real_batch.numpy() + 1.0) / 2.0).flatten()
dcgan_pixels = dcgan_16.numpy().flatten()
lsgan_pixels = lsgan_16.numpy().flatten()

plt.figure(figsize=(10, 5))
plt.hist(real_pixels, bins=50, density=True, alpha=0.5, color='black', label='Реальные изображения')
plt.hist(dcgan_pixels, bins=50, density=True, alpha=0.5, color='blue', label='DCGAN сгенерированные')
plt.hist(lsgan_pixels, bins=50, density=True, alpha=0.5, color='green', label='LSGAN сгенерированные')
plt.title('Распределение попиксельной интенсивности яркости (Real vs DCGAN vs LSGAN)', fontsize=12)
plt.xlabel('Нормализованная яркость пикселя [0, 1]')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '10_pixel_intensity_distribution.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 5. Вычисление метрик качества генерации (Fréchet Distance и статистика)
# ------------------------------------------------------------------------------
def calculate_activation_statistics(images_tensor):
    # Преобразование изображений в плоские векторы признаков
    flat_features = images_tensor.view(images_tensor.size(0), -1).numpy()
    mu = np.mean(flat_features, axis=0)
    sigma = np.cov(flat_features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        covmean = np.identity(len(diff)) * eps
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

real_imgs_all = []
for i, (imgs, _) in enumerate(dataloader):
    real_imgs_all.append((imgs + 1.0) / 2.0)
    if len(real_imgs_all) >= 8:  # ~1024 изображения для оценки
        break
real_eval = torch.cat(real_imgs_all, dim=0)[:1000]

with torch.no_grad():
    eval_noise = torch.randn(1000, latent_dim, device=device)
    dcgan_eval = ((dcgan_G(eval_noise).cpu() + 1.0) / 2.0)
    lsgan_eval = ((lsgan_G(eval_noise).cpu() + 1.0) / 2.0)

mu_real, sigma_real = calculate_activation_statistics(real_eval)
mu_dcgan, sigma_dcgan = calculate_activation_statistics(dcgan_eval)
mu_lsgan, sigma_lsgan = calculate_activation_statistics(lsgan_eval)

fd_dcgan = calculate_frechet_distance(mu_real, sigma_real, mu_dcgan, sigma_dcgan)
fd_lsgan = calculate_frechet_distance(mu_real, sigma_real, mu_lsgan, sigma_lsgan)

print("\n" + "=" * 75)
print("МЕТРИКИ КАЧЕСТВА ГЕНЕРАЦИИ (FRÉCHET DISTANCE):")
print(f"Fréchet Distance (DCGAN): {fd_dcgan:.2f}")
print(f"Fréchet Distance (LSGAN): {fd_lsgan:.2f}")
print("=" * 75)

# График 11: Сравнение качества моделей
models = ['DCGAN (BCE Loss)', 'LSGAN (MSE Loss)']
fd_values = [fd_dcgan, fd_lsgan]
final_d_acc = [(dcgan_hist['d_real_acc'][-1] + dcgan_hist['d_fake_acc'][-1])/2.0,
               (lsgan_hist['d_real_acc'][-1] + lsgan_hist['d_fake_acc'][-1])/2.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
bars1 = ax1.bar(models, fd_values, color=['royalblue', 'forestgreen'], width=0.5)
ax1.set_title('Fréchet Distance (чем ниже, тем ближе к реальным)', fontsize=11)
ax1.set_ylabel('Расстояние Фреше (FD)')
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

bars2 = ax2.bar(models, final_d_acc, color=['coral', 'mediumpurple'], width=0.5)
ax2.set_title('Финальная точность Дискриминатора (D-Accuracy, %)', fontsize=11)
ax2.set_ylabel('Точность (%)')
ax2.set_ylim(0, 100)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Сводное сравнение метрик качества генеративных моделей', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '11_generation_quality_metrics.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 6. Сводная таблица результатов
# ------------------------------------------------------------------------------
summary_df = pd.DataFrame({
    'Архитектура GAN': ['DCGAN (BCE Loss)', 'LSGAN (MSE Loss)'],
    'Финальный Loss D': [dcgan_hist['loss_D'][-1], lsgan_hist['loss_D'][-1]],
    'Финальный Loss G': [dcgan_hist['loss_G'][-1], lsgan_hist['loss_G'][-1]],
    'D Real Accuracy (%)': [dcgan_hist['d_real_acc'][-1], lsgan_hist['d_real_acc'][-1]],
    'D Fake Accuracy (%)': [dcgan_hist['d_fake_acc'][-1], lsgan_hist['d_fake_acc'][-1]],
    'Fréchet Distance': [fd_dcgan, fd_lsgan]
})
print("\n" + "=" * 75)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТА:")
print(summary_df.to_string(index=False))
print("=" * 75)
print(f"Все 11 графиков успешно сохранены в директорию: {plots_dir}")
