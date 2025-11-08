# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] ALS 모델에 대한 다차원 심층 성능 평가 및 사례 연구
# [End-to-End 단계] 6. 평가(Advanced Evaluation & Case Study)
#
# (이전 설명과 동일)
# ...
# [추가된 내용]
#  - 정량적 평가 지표(Precision, Similarity) 계산 후, 무작위로 선정된 5명의 실제
#    추천 사례를 질적으로 분석하여 모델의 행동을 직관적으로 이해합니다.
# ======================================================================================

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import random # 💥💥💥 랜덤 샘플링을 위해 임포트 💥💥💥

# --- 1. 설정 ---
DATA_PATH = '../../data/final-lastfm-data.csv'
K_FOR_RANKING = 10
Eth = 10 
factors = 50
regularization = 0.01
iterations = 20
random_state = 42

# --- 2. 데이터 준비 ---
df_raw = pd.read_csv(DATA_PATH)
user_counts = df_raw['user_id'].value_counts()
active_users = user_counts[user_counts >= Eth].index
df = df_raw[df_raw['user_id'].isin(active_users)].copy()
unique_users = df['user_id'].unique()
unique_items = df['title'].unique()
user_to_idx = {user: i for i, user in enumerate(unique_users)}
item_to_idx = {item: i for i, item in enumerate(unique_items)}
idx_to_user = {i: user for user, i in user_to_idx.items()}
idx_to_item = {i: item for item, i in item_to_idx.items()}
num_users = len(unique_users)
num_items = len(unique_items)

# --- 3. Train / Test 데이터 분할 ---
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['user_id'], random_state=41
)
train_user_indices = train_df['user_id'].map(user_to_idx)
train_item_indices = train_df['title'].map(item_to_idx)
train_matrix = csr_matrix((train_df['play_count'].astype(float),
                           (train_user_indices, train_item_indices)),
                          shape=(num_users, num_items))
train_item_user = train_matrix.T.tocsr()

# --- 4. ALS 모델 학습 ---
model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                iterations=iterations, random_state=random_state)
model.fit(train_item_user, show_progress=True)


# --- 5. 다차원 성능 평가 로직 ---
# (생략 - 이전 코드와 동일하게 실행하여 avg_precision, avg_recall, avg_genre_similarity 계산)
song_features_df = df.drop_duplicates(subset='title').set_index('title')
main_genres = ['Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 'Folk & Country', 'Reggae', 'Other']
existing_genre_cols = [col for col in main_genres if col in song_features_df.columns]
song_features_matrix_map = {title: features for title, features in zip(song_features_df.index, song_features_df[existing_genre_cols].values)}
precisions, recalls, all_users_avg_similarities = [], [], []
true_relevants = test_df.groupby('user_id')['title'].apply(set).to_dict()
test_users_indices = [user_to_idx[user] for user in true_relevants.keys() if user in user_to_idx]
# 💥💥💥 사례 연구를 위해 사용자별 추천 결과를 저장할 딕셔너리 추가 💥💥💥
case_study_results = {}

for user_idx in tqdm(test_users_indices, desc="Evaluating"):
    user_id_str = idx_to_user[user_idx]
    recs_indices, _ = model.recommend(
        userid=user_idx, user_items=train_matrix[user_idx],
        N=K_FOR_RANKING, filter_already_liked_items=False
    )
    already_liked_indices = set(train_matrix[user_idx].indices)
    filtered_recs_indices = [idx for idx in recs_indices if idx not in already_liked_indices]
    recommended_items = {idx_to_item[idx] for idx in filtered_recs_indices}
    
    # (Precision/Recall, Genre Similarity 계산 로직은 이전과 동일 - 생략)
    ground_truth_items = true_relevants.get(user_id_str, set())
    if ground_truth_items:
        intersection = recommended_items.intersection(ground_truth_items)
        precisions.append(len(intersection) / K_FOR_RANKING)
        recalls.append(len(intersection) / len(ground_truth_items))
    user_train_data = train_df[train_df['user_id'] == user_id_str]
    if not user_train_data.empty:
        user_genre_data = user_train_data[existing_genre_cols]
        play_counts = user_train_data['play_count']
        user_profile_vector = (user_genre_data.mul(play_counts, axis=0).sum() / play_counts.sum()).values.reshape(1, -1)
        current_user_similarities = []
        for song_title in recommended_items:
            if song_title in song_features_matrix_map:
                song_vector = song_features_matrix_map[song_title].reshape(1, -1)
                similarity = cosine_similarity(user_profile_vector, song_vector)[0][0]
                current_user_similarities.append(similarity)
        if current_user_similarities:
            all_users_avg_similarities.append(np.mean(current_user_similarities))

    # 💥💥💥 [사례 연구] 현재 사용자의 분석 결과를 딕셔너리에 저장 💥💥💥
    case_study_results[user_id_str] = {
        'profile_vector': user_profile_vector.flatten(),
        'recommendations': recommended_items
    }

avg_precision = np.mean(precisions) if precisions else 0
avg_recall = np.mean(recalls) if recalls else 0
f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) != 0 else 0
avg_genre_similarity = np.mean(all_users_avg_similarities) if all_users_avg_similarities else 0


# --- 6. 최종 결과 출력 ---
print("\n" + "="*60)
print(f"▶ ALS 모델 심층 성능 평가 요약 (MSD, sklearn split)")
print("="*60)
print(f"\n[1. Top-{K_FOR_RANKING} 추천 성능 지표 (정답률)]\n")
print(f"Precision@{K_FOR_RANKING}: {avg_precision:.4f}")
print(f"Recall@{K_FOR_RANKING}   : {avg_recall:.4f}")
print(f"F1-Score@{K_FOR_RANKING}  : {f1_score:.4f}")
print("\n" + "-"*60)
print(f"\n[2. Top-{K_FOR_RANKING} 콘텐츠 정렬 성능 (취향 일치도)]\n")
print(f"Average Genre Similarity: {avg_genre_similarity:.4f}")
print("="*60)


# --- 7. 💥💥💥 사례 연구 (Case Study) 결과 출력 💥💥💥 ---
print("\n" + "="*70)
print("▶ 사례 연구: 랜덤 사용자 5명에 대한 심층 분석")
print("="*70)

# 분석 대상 사용자 ID 목록에서 5명을 무작위로 추출
# 만약 5명보다 적으면, 가능한 만큼만 추출
num_samples = min(5, len(case_study_results))
random_users = random.sample(list(case_study_results.keys()), num_samples)

for i, user_id in enumerate(random_users):
    user_data = case_study_results[user_id]
    profile_vector = user_data['profile_vector']
    recommendations = user_data['recommendations']
    
    print(f"\n--- [Case {i+1}] User: {user_id} ---\n")
    
    # 1. 사용자의 Top 5 선호 장르 출력
    # 취향 프로필 벡터를 (장르, 점수) 쌍으로 만든 후, 점수가 높은 순으로 정렬
    user_top_genres = sorted(zip(existing_genre_cols, profile_vector), key=lambda x: x[1], reverse=True)
    
    print("  [사용자 주요 취향 (Top 5 Genres)]")
    for genre, score in user_top_genres[:5]:
        # 점수가 0.01보다 큰 경우에만 의미 있는 취향으로 간주하여 출력
        if score > 0.01:
            print(f"    - {genre:<25} ({score:.2%})") # 퍼센트로 보기 좋게 출력
            
    # 2. 추천된 노래 목록과 각 노래의 Top 3 장르 출력
    print("\n  [ALS 추천 목록 (Top K Songs & Genres)]")
    if not recommendations:
        print("    (추천된 노래가 없습니다)")
    else:
        for song_title in recommendations:
            print(f"    - {song_title}")
            # 추천된 노래의 장르 벡터를 가져와서, 점수가 높은 순으로 정렬
            if song_title in song_features_matrix_map:
                song_vector = song_features_matrix_map[song_title]
                song_top_genres = sorted(zip(existing_genre_cols, song_vector), key=lambda x: x[1], reverse=True)
                
                # 해당 노래의 주요 장르(점수가 0.01 이상인)를 최대 3개까지 출력
                genre_str = ", ".join([f"{g}" for g, s in song_top_genres[:5] if s > 0.01])
                print(f"      └─ Genres: {genre_str if genre_str else 'N/A'}")

print("\n" + "="*70)