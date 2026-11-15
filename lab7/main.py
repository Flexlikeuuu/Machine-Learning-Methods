import os
import shutil
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 0. Настройка директории для графиков
# ------------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_dir, 'plots')
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)
os.makedirs(plots_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Загрузка данных и предварительный анализ
# ------------------------------------------------------------------------------
data_path = os.path.join(base_dir, 'wine-clustering.csv')
raw_df = pd.read_csv(data_path)

print("=" * 70)
print("1. ИСХОДНЫЕ ДАННЫЕ WINE-CLUSTERING.CSV")
print("=" * 70)
print(f"Исходная размерность набора данных: {raw_df.shape[0]} строк, {raw_df.shape[1]} столбцов")
print("\nПропущенные значения по признакам:")
print(raw_df.isna().sum())
print("\nБазовые статистические характеристики:")
print(raw_df.describe().round(2).to_string())

feature_names = raw_df.columns.tolist()

# ------------------------------------------------------------------------------
# 2. Очистка от выбросов (метод 1.5 * IQR) и стандартизация
# ------------------------------------------------------------------------------
# Диаграммы размаха до очистки
plt.figure(figsize=(15, 9))
sns.set_theme(style='whitegrid', font_scale=0.9)
for idx, col in enumerate(feature_names, 1):
    plt.subplot(3, 5, idx)
    sns.boxplot(y=raw_df[col], color='#5dade2', flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'red'})
    plt.title(col, fontsize=10, fontweight='bold')
    plt.ylabel('')
plt.suptitle('Диаграммы размаха признаков до очистки от выбросов', fontsize=14, y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_boxplots_before_outliers.png'), dpi=300)
plt.close()

# Вычисление границ IQR
q1 = raw_df.quantile(0.25)
q3 = raw_df.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = ((raw_df < lower_bound) | (raw_df > upper_bound)).any(axis=1)
outlier_count = outlier_mask.sum()
df_clean = raw_df[~outlier_mask].copy().reset_index(drop=True)

print("\n" + "=" * 70)
print("2. ПРЕДОБРАБОТКА И ОЧИСТКА ВЫБРОСОВ (МЕТОД IQR)")
print("=" * 70)
print(f"Количество обнаруженных наблюдений с выбросами (1.5*IQR): {outlier_count}")
print(f"Размерность очищенного набора данных: {df_clean.shape[0]} строк, {df_clean.shape[1]} признаков")

# Диаграммы размаха после очистки
plt.figure(figsize=(15, 9))
for idx, col in enumerate(feature_names, 1):
    plt.subplot(3, 5, idx)
    sns.boxplot(y=df_clean[col], color='#58d68d')
    plt.title(col, fontsize=10, fontweight='bold')
    plt.ylabel('')
plt.suptitle('Диаграммы размаха признаков после фильтрации выбросов (1.5·IQR)', fontsize=14, y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_boxplots_after_outliers.png'), dpi=300)
plt.close()

# Стандартизация признаков (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Матрица корреляций
plt.figure(figsize=(11, 9))
corr = df_clean.corr()
mask_upper = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask_upper, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True,
            cbar_kws={'shrink': 0.8})
plt.title('Корреляционная матрица признаков химического состава вин', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_correlation_matrix.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 3. Понижение размерности: PCA и t-SNE
# ------------------------------------------------------------------------------
pca_full = PCA().fit(X_scaled)
exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

print("\n" + "=" * 70)
print("3. АНАЛИЗ ГЛАВНЫХ КОМПОНЕНТ (PCA)")
print("=" * 70)
for i in range(len(exp_var)):
    print(f"PC{i+1}: доля объясненной дисперсии = {exp_var[i]:.4f} ({exp_var[i]*100:.1f}%), накопленная = {cum_var[i]:.4f} ({cum_var[i]*100:.1f}%)")

plt.figure(figsize=(9, 5))
x_axis = range(1, len(exp_var) + 1)
plt.bar(x_axis, exp_var * 100, alpha=0.7, color='steelblue', label='Индивидуальная дисперсия компоненты')
plt.step(x_axis, cum_var * 100, where='mid', color='crimson', lw=2, label='Кумулятивная дисперсия')
plt.axhline(y=70, color='gray', linestyle='--', label='Порог 70% дисперсии')
plt.xlabel('Номер главной компоненты')
plt.ylabel('Доля объясненной дисперсии (%)')
plt.title('График каменистой осыпи (Scree Plot) и накопленная дисперсия PCA')
plt.xticks(x_axis)
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_pca_variance_explained.png'), dpi=300)
plt.close()

# Проекция на 2 главных компоненты
pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_scaled)

# Нелинейное проецирование через t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# ------------------------------------------------------------------------------
# 4. Модель 1: K-Means и подбор оптимального k
# ------------------------------------------------------------------------------
k_range = range(2, 11)
wcss = []
sil_kmeans = []
db_kmeans = []
ch_kmeans = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=15, max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_kmeans.append(silhouette_score(X_scaled, labels))
    db_kmeans.append(davies_bouldin_score(X_scaled, labels))
    ch_kmeans.append(calinski_harabasz_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(k_range, wcss, marker='o', lw=2, color='#2980b9')
ax1.axvline(x=3, color='crimson', linestyle='--', label='Оптимум k=3 (локоть)')
ax1.set_title('Метод локтя (Elbow Method / WCSS)')
ax1.set_xlabel('Количество кластеров (k)')
ax1.set_ylabel('Инерция (Within-Cluster Sum of Squares)')
ax1.set_xticks(k_range)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, sil_kmeans, marker='s', lw=2, color='#27ae60', label='Silhouette Score')
ax2.axvline(x=3, color='crimson', linestyle='--', label='Максимум при k=3')
ax2.set_title('Коэффициент силуэта (Silhouette Score) vs k')
ax2.set_xlabel('Количество кластеров (k)')
ax2.set_ylabel('Средний коэффициент силуэта')
ax2.set_xticks(k_range)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_kmeans_elbow_silhouette.png'), dpi=300)
plt.close()

# Обучение итоговой модели K-Means с k=3
best_k = 3
kmeans_final = KMeans(n_clusters=best_k, init='k-means++', n_init=20, random_state=42)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

# Анализ силуэтных коэффициентов по кластерам для K-Means
sample_silhouette_values = silhouette_samples(X_scaled, kmeans_labels)
plt.figure(figsize=(8, 6))
y_lower = 10
colors = ['#e74c3c', '#2ecc71', '#3498db']
for i in range(best_k):
    ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = colors[i]
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Кластер {i}', fontsize=10, fontweight='bold')
    y_lower = y_upper + 10

avg_sil = silhouette_score(X_scaled, kmeans_labels)
plt.axvline(x=avg_sil, color='black', linestyle='--', label=f'Средний силуэт = {avg_sil:.3f}')
plt.title('Силуэтная диаграмма для K-Means (k=3)', fontsize=12)
plt.xlabel('Коэффициент силуэта')
plt.ylabel('Элементы кластеров')
plt.yticks([])
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '06_kmeans_silhouette_samples.png'), dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 5. Модель 2: Иерархическая кластеризация (Agglomerative Clustering)
# ------------------------------------------------------------------------------
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, no_labels=True)
plt.axhline(y=12.5, color='crimson', linestyle='--', lw=2, label='Порог отсечения (3 кластера)')
plt.title('Дендрограмма агломеративной иерархической кластеризации (метод Уорда)', fontsize=12)
plt.xlabel('Индексы объектов выборки')
plt.ylabel('Евклидово расстояние объединения')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '07_hierarchical_dendrogram.png'), dpi=300)
plt.close()

agg_final = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_final.fit_predict(X_scaled)

# ------------------------------------------------------------------------------
# 6. Модель 3: Плотностная кластеризация DBSCAN
# ------------------------------------------------------------------------------
# k-distance graph для определения epsilon
k_neighbors = 5
nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
sorted_distances = np.sort(distances[:, k_neighbors - 1])

plt.figure(figsize=(8, 5))
plt.plot(sorted_distances, lw=2, color='#8e44ad')
plt.axhline(y=2.35, color='crimson', linestyle='--', label='Выбранный эпсилон (ε = 2.35)')
plt.title('График расстояний до 5-го ближайшего соседа (k-distance graph)', fontsize=12)
plt.xlabel('Индекс объекта (отсортировано по возрастанию)')
plt.ylabel('Расстояние до 5-го соседа')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '08_dbscan_k_distance.png'), dpi=300)
plt.close()

# Подбор параметров DBSCAN
best_dbscan_sil = -1
best_eps = 2.35
best_ms = 4
best_dbscan_labels = None

for eps in np.arange(1.8, 3.0, 0.15):
    for ms in range(3, 7):
        db = DBSCAN(eps=eps, min_samples=ms).fit(X_scaled)
        n_c = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        if n_c >= 2:
            mask = db.labels_ != -1
            if mask.sum() > n_c:
                s = silhouette_score(X_scaled[mask], db.labels_[mask])
                if s > best_dbscan_sil and (db.labels_ == -1).sum() < len(X_scaled) * 0.35:
                    best_dbscan_sil = s
                    best_eps = eps
                    best_ms = ms
                    best_dbscan_labels = db.labels_

if best_dbscan_labels is None:
    db_final = DBSCAN(eps=2.35, min_samples=4).fit(X_scaled)
    best_dbscan_labels = db_final.labels_
else:
    db_final = DBSCAN(eps=best_eps, min_samples=best_ms).fit(X_scaled)
    best_dbscan_labels = db_final.labels_

dbscan_labels = best_dbscan_labels

# ------------------------------------------------------------------------------
# 7. Модель 4: Вероятностная модель Gaussian Mixture Model (GMM)
# ------------------------------------------------------------------------------
bic_scores = []
aic_scores = []
gmm_range = range(2, 9)

for n in gmm_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

plt.figure(figsize=(8, 5))
plt.plot(gmm_range, bic_scores, marker='o', lw=2, color='#d35400', label='BIC (Байесовский критерий)')
plt.plot(gmm_range, aic_scores, marker='s', lw=2, color='#16a085', label='AIC (Критерий Акаике)')
plt.axvline(x=3, color='crimson', linestyle='--', label='Оптимум n_components = 3')
plt.title('Критерии BIC и AIC для модели смеси гауссиан (GMM)', fontsize=12)
plt.xlabel('Количество компонент')
plt.ylabel('Значение критерия')
plt.xticks(gmm_range)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '09_gmm_bic_aic.png'), dpi=300)
plt.close()

gmm_final = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_labels = gmm_final.fit_predict(X_scaled)

# ------------------------------------------------------------------------------
# 8. Сводная оценка качества моделей
# ------------------------------------------------------------------------------
models_dict = {
    'K-Means (k=3)': kmeans_labels,
    'Agglomerative (k=3, Ward)': agg_labels,
    'DBSCAN (eps=2.35, ms=4)': dbscan_labels,
    'GMM (k=3)': gmm_labels
}

results = []
for name, labels in models_dict.items():
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = (labels == -1).sum()
    if n_clusters > 1 and mask.sum() > n_clusters:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        db = davies_bouldin_score(X_scaled[mask], labels[mask])
        ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
    else:
        sil, db, ch = np.nan, np.nan, np.nan
    results.append({
        'Model': name,
        'Clusters': n_clusters,
        'Noise Points': noise_count,
        'Silhouette Score': sil,
        'Davies-Bouldin': db,
        'Calinski-Harabasz': ch
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("4. СВОДНАЯ ТАБЛИЦА МЕТРИК КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
print("=" * 70)
print(results_df.to_string(index=False))

# ------------------------------------------------------------------------------
# 9. Визуализация кластеров: PCA 2D и t-SNE 2D
# ------------------------------------------------------------------------------
def plot_multi_clustering(X_proj, xlabel, ylabel, title_prefix, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    for idx, (m_name, labels) in enumerate(models_dict.items()):
        ax = axes[idx]
        unique_labels = sorted(set(labels))
        palette = sns.color_palette('tab10', len(unique_labels))
        for u_idx, u_lbl in enumerate(unique_labels):
            u_mask = labels == u_lbl
            if u_lbl == -1:
                ax.scatter(X_proj[u_mask, 0], X_proj[u_mask, 1], c='gray', marker='x', s=35, label='Шум (-1)', alpha=0.6)
            else:
                ax.scatter(X_proj[u_mask, 0], X_proj[u_mask, 1], color=palette[u_idx], s=40, label=f'Кластер {u_lbl}', alpha=0.85)
        ax.set_title(f'{m_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'{title_prefix}', fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close()

plot_multi_clustering(X_pca, f'PC1 ({exp_var[0]*100:.1f}%)', f'PC2 ({exp_var[1]*100:.1f}%)',
                      'Сравнение алгоритмов кластеризации в проекции PCA (PC1 vs PC2)', '10_clusters_pca_comparison.png')

plot_multi_clustering(X_tsne, 't-SNE Измерение 1', 't-SNE Измерение 2',
                      'Сравнение алгоритмов кластеризации в нелинейной проекции t-SNE', '11_clusters_tsne_comparison.png')

# ------------------------------------------------------------------------------
# 10. Профиль химических признаков кластеров (K-Means)
# ------------------------------------------------------------------------------
df_profile = pd.DataFrame(X_scaled, columns=feature_names)
df_profile['Cluster'] = kmeans_labels
cluster_means = df_profile.groupby('Cluster').mean()

plt.figure(figsize=(13, 5))
sns.heatmap(cluster_means, annot=True, fmt='.2f', cmap='vlag', center=0, cbar_kws={'label': 'Z-score среднего значения'})
plt.title('Профиль центроидов кластеров (K-Means): средние стандартизированные значения признаков', fontsize=12, pad=12)
plt.xlabel('Химические признаки')
plt.ylabel('Идентификатор кластера')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '12_cluster_feature_heatmap.png'), dpi=300)
plt.close()

print(f"\nВсе 12 графиков успешно сгенерированы и сохранены в директорию: {plots_dir}")
