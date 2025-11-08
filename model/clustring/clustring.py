# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] K-Means 클러스터링을 이용한 사용자 취향 그룹 분석 (User Segmentation)
# [End-to-End 단계] 4. 탐색적 데이터 분석 (User Analysis)
# ======================================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. 설정 ---
FINAL_FILTERED_CSV = '../../data/final-lastfm-data.csv' 
# -----------------


# --- 2. 💥💥💥 수정된 데이터 준비 및 사용자 프로필 벡터 생성 💥💥💥 ---
try:
    print(f"'{FINAL_FILTERED_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(FINAL_FILTERED_CSV)
    print("-> 로딩 완료.")
except FileNotFoundError:
    print(f"오류: '{FINAL_FILTERED_CSV}' 파일을 찾을 수 없습니다.")
    exit()

main_genres = ['Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 'Folk & Country', 'Reggae', 'Other']
existing_genre_cols = [col for col in main_genres if col in df.columns]

# [핵심 수정 1] 결과를 저장할 빈 딕셔너리를 생성합니다.
# Key: user_id, Value: user_profile_vector
user_profiles_dict = {}

for user_id, user_data in tqdm(df.groupby('user_id'), desc="Building user profiles"):
    play_counts = user_data['play_count']
    # play_counts의 합이 0인 경우를 방지하여 0으로 나누는 오류를 막습니다.
    if play_counts.sum() > 0:
        user_profile_vector = (user_data[existing_genre_cols].mul(play_counts, axis=0).sum() / play_counts.sum()).values
        # [핵심 수정 2] 딕셔너리에 '사용자 ID'를 key로, '취향 벡터'를 value로 저장합니다.
        user_profiles_dict[user_id] = user_profile_vector

# [핵심 수정 3] 딕셔너리로부터 직접 Pandas DataFrame을 생성합니다.
# orient='index'는 딕셔너리의 key를 DataFrame의 인덱스로 사용하라는 의미입니다.
user_profile_df = pd.DataFrame.from_dict(
    user_profiles_dict, 
    orient='index', 
    columns=existing_genre_cols
)

# 클러스터링에 사용할 순수 장르 데이터
user_genre_matrix = user_profile_df[existing_genre_cols]

print(f"\n클러스터링 분석 대상 고유 사용자 수: {len(user_genre_matrix)}명")
# -----------------


# --- 3. 최적의 군집 수(K) 찾기 (Elbow Method) ---
# (이하 코드는 이전과 동일하며, 이제 정상적으로 작동합니다.)
print("\n최적의 사용자 군집 수(K)를 찾기 위해 엘보우 기법을 실행합니다...")
inertia_values = []
possible_k_values = range(2, 16)

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(user_genre_matrix)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.plot(possible_k_values, inertia_values, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K (User Clusters)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(possible_k_values)
plt.grid(True)
plt.savefig('user_clusters_elbow_method.png', dpi=300, bbox_inches='tight')
print("-> Elbow Method 그래프를 'user_clusters_elbow_method.png' 파일로 저장했습니다.")
plt.show()
# -----------------


# --- 4. K-Means 모델 학습 및 사용자 군집 할당 ---
try:
    OPTIMAL_K = int(input("\n그래프를 보고 최적의 K값을 입력하세요: "))
except (ValueError, EOFError):
    print("잘못된 입력입니다. 기본값인 5로 설정합니다.")
    OPTIMAL_K = 5

print(f"\n선택된 K={OPTIMAL_K} 값으로 최종 K-Means 모델을 학습합니다...")
kmeans_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto')
cluster_labels = kmeans_final.fit_predict(user_genre_matrix)
user_genre_matrix['cluster'] = cluster_labels
print("-> 모든 사용자에게 군집 할당 완료.")
# -----------------


# --- 5. 결과 분석 및 히트맵 저장 ---
print("\n" + "="*60)
print(f"▶ 각 사용자 군집의 장르적 특성 분석 (K={OPTIMAL_K})")
print("="*60)

print("\n[1. 군집별 사용자 수]\n")
print(user_genre_matrix['cluster'].value_counts().sort_index())

cluster_centers = user_genre_matrix.groupby('cluster')[existing_genre_cols].mean()
print("\n[2. 군집별 평균 장르 취향 (군집의 '성격')]\n")
pd.options.display.float_format = '{:.3f}'.format
print(cluster_centers)

plt.figure(figsize=(16, 8))
sns.heatmap(cluster_centers, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
plt.title(f'Genre Preferences of Each User Cluster (K={OPTIMAL_K})', fontsize=16)
plt.xlabel('Genres', fontsize=12)
plt.ylabel('User Cluster ID', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig('user_cluster_characteristics.png', dpi=300, bbox_inches='tight')
print("\n-> 사용자 군집 특성 히트맵을 'user_cluster_characteristics.png' 파일로 저장했습니다.")
plt.show()
# -----------------


# --- 6. 군집 분포 시각화 (t-SNE) ---
print("\n▶ 사용자 군집 분포 시각화 (t-SNE 적용)")
X = user_genre_matrix[existing_genre_cols]

print("\nt-SNE를 사용하여 고차원 -> 2차원으로 축소 중...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)
df_tsne = pd.DataFrame(X_tsne, columns=['tsne_x', 'tsne_y'])
# 💥💥💥 인덱스를 기준으로 cluster 라벨을 매칭합니다 💥💥💥
df_tsne['cluster'] = user_genre_matrix['cluster'].values

plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=df_tsne,
    x='tsne_x',
    y='tsne_y',
    hue='cluster',
    palette=sns.color_palette("hsv", OPTIMAL_K),
    legend='full',
    alpha=0.6
)
plt.title('Lastfm User Clusters Visualization using t-SNE', fontsize=18)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True)
plt.savefig('Lastfm user_cluster_distribution.png', dpi=300, bbox_inches='tight')
print("-> 사용자 군집 분포도를 'user_cluster_distribution.png' 파일로 저장했습니다.")
plt.show()
# ======================================================================================