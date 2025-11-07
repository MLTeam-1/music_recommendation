# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] K-Means 클러스터링 분석 및 결과 시각화/저장
# [End-to-End 단계] 4. 탐색적 데이터 분석 (Exploratory Data Analysis)
#
# (이전 설명과 동일)
# ...
# [추가된 내용]
#  5. 차원 축소 (PCA, t-SNE)를 이용해 고차원의 장르 데이터를 2차원 평면에 시각화하여,
#     각 노래가 어떻게 군집을 형성하는지 분포를 확인하고 PNG 파일로 저장합니다.
# ======================================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA       # 💥💥💥 PCA 임포트 💥💥💥
from sklearn.manifold import TSNE         # 💥💥💥 t-SNE 임포트 💥💥💥
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 설정 ---
FINAL_FILTERED_CSV = '../../data/final-msd-data.csv' 

# --- 2. 데이터 준비 ---
try:
    df = pd.read_csv(FINAL_FILTERED_CSV)
except FileNotFoundError:
    print(f"오류: '{FINAL_FILTERED_CSV}' 파일을 찾을 수 없습니다.")
    exit()

song_features_df = df.drop_duplicates(subset='title').set_index('title')
main_genres = ['Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 'Folk & Country', 'Reggae', 'Other']
existing_genre_cols = [col for col in main_genres if col in song_features_df.columns]
song_genre_matrix = song_features_df[existing_genre_cols]
song_genre_matrix = song_genre_matrix[song_genre_matrix.sum(axis=1) > 0]

# --- 3. 최적 K 찾기 (Elbow Method) ---
# (생략 - 이전 코드와 동일하게 실행하여 그래프 확인 후 K값 결정)
OPTIMAL_K = 11 

# --- 4. K-Means 모델 학습 ---
kmeans_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto')
cluster_labels = kmeans_final.fit_predict(song_genre_matrix)
song_genre_matrix['cluster'] = cluster_labels

# --- 5. 결과 분석 및 히트맵 저장 ---
# (생략 - 이전 코드와 동일하게 실행하여 히트맵 생성 및 저장)
cluster_centers = song_genre_matrix.groupby('cluster')[existing_genre_cols].mean()


# 💥💥💥 --- 6. 군집 분포 시각화 (차원 축소) 및 PNG 저장 --- 💥💥💥
print("\n" + "="*60)
print(f"▶ 군집 분포 시각화 (차원 축소 기법 적용)")
print("="*60)

# 클러스터링에 사용된 순수 장르 데이터 (cluster 컬럼 제외)
X = song_genre_matrix[existing_genre_cols]

# [방법 1] PCA를 이용한 차원 축소
print("\n[1] PCA를 사용하여 12차원 -> 2차원으로 축소 중...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
# 결과를 DataFrame으로 만들어 시각화 준비
df_pca = pd.DataFrame(X_pca, columns=['pca_x', 'pca_y'])
df_pca['cluster'] = cluster_labels
df_pca['title'] = X.index

# [방법 2] t-SNE를 이용한 차원 축소 (시간이 더 걸릴 수 있음)
print("\n[2] t-SNE를 사용하여 12차원 -> 2차원으로 축소 중... (시간이 다소 소요될 수 있습니다)")
# perplexity: t-SNE의 중요 파라미터. 보통 5~50 사이의 값을 사용.
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(X)
# 결과를 DataFrame으로 만들어 시각화 준비
df_tsne = pd.DataFrame(X_tsne, columns=['tsne_x', 'tsne_y'])
df_tsne['cluster'] = cluster_labels
df_tsne['title'] = X.index

# [시각화 및 저장]
print("\n[3] 축소된 데이터를 바탕으로 군집 분포도 시각화 및 파일 저장...")

# 전체 그림판(figure)을 2개의 하위 그래프(axes)로 나눔
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# 첫 번째 그래프: PCA 결과
sns.scatterplot(
    data=df_pca,
    x='pca_x',
    y='pca_y',
    hue='cluster', # 'cluster' 컬럼 값에 따라 점의 색깔을 다르게 함
    palette=sns.color_palette("hsv", OPTIMAL_K), # K개의 고유한 색상 팔레트 사용
    legend='full',
    alpha=0.6, # 점의 투명도
    ax=ax1
)
ax1.set_title('Song Clusters Visualization using PCA', fontsize=18)
ax1.set_xlabel('Principal Component 1', fontsize=12)
ax1.set_ylabel('Principal Component 2', fontsize=12)
ax1.grid(True)

# 두 번째 그래프: t-SNE 결과
sns.scatterplot(
    data=df_tsne,
    x='tsne_x',
    y='tsne_y',
    hue='cluster',
    palette=sns.color_palette("hsv", OPTIMAL_K),
    legend='full',
    alpha=0.6,
    ax=ax2
)
ax2.set_title('Song Clusters Visualization using t-SNE', fontsize=18)
ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax2.grid(True)

# 최종 그림을 파일로 저장
plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
print("-> 군집 분포도를 'cluster_distribution.png' 파일로 저장했습니다.")
plt.show()
# ======================================================================================