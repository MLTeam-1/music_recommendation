# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] SVD 모델에 대한 다차원 심층 성능 평가 및 사례 연구
# [End-to-End 단계] 6. 평가(Advanced Evaluation & Case Study)
# ======================================================================================

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import random

# Surprise 라이브러리
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# --- 1. 설정 ---
FINAL_MERGED_CSV = '../../data/final-lastfm-data.csv'
K = 10
Eth = 10
# -----------------


# --- 2. 데이터 로딩 및 전처리 ---
try:
    df = pd.read_csv(FINAL_MERGED_CSV)
except FileNotFoundError:
    print(f"오류: '{FINAL_MERGED_CSV}' 파일을 찾을 수 없습니다.")
    exit()

print(f"청취횟수가 {Eth}회 이상인 유저만 포함")
user_counts = df['user_id'].value_counts()
active_users = user_counts[user_counts >= Eth].index
df = df[df['user_id'].isin(active_users)]
print(f"전체 데이터 수 : {df['user_id'].count()}")
df['rating'] = np.log1p(df['play_count'])
# -----------------


# --- 3. Surprise 데이터셋 생성 및 분리 ---
reader = Reader(rating_scale=(df['play_count'].min(), df['play_count'].max()))
data = Dataset.load_from_df(df[['user_id', 'title', 'play_count']], reader)
reader_log = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
data_log = Dataset.load_from_df(df[['user_id', 'title', 'rating']], reader_log)

print("\n데이터를 Train Set (80%)과 Test Set (20%)으로 분리합니다...")
trainset, testset = train_test_split(data, test_size=0.2, random_state=41)
trainset_log, testset_log = train_test_split(data_log, test_size=0.2, random_state=41)
print("-> 데이터 분리 완료.")
# -----------------


# --- 4. 모델 학습 ---
print("\nTrain Set을 사용하여 SVD 모델을 학습합니다...")
algo_for_evaluation = SVD(n_factors=100, n_epochs=20, random_state=42)
algo_for_evaluation_log = SVD(n_factors=100, n_epochs=20, random_state=42)
algo_for_evaluation.fit(trainset)
algo_for_evaluation_log.fit(trainset_log)
print("-> 모델 학습 완료.")
# -----------------


# --- 5. 평가 함수 및 데이터 준비 ---
def precision_recall_at_k_implicit(predictions, k=10):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions, recalls = dict(), dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r > 0) for (_, true_r) in user_ratings)
        n_rel_and_rec_k = sum(((true_r > 0)) for (_, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return sum(p for p in precisions.values()) / len(precisions), sum(r for r in recalls.values()) / len(recalls)

song_features_df = df.drop_duplicates(subset='title').set_index('title')
main_genres = ['Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 'Folk & Country', 'Reggae', 'Other']
existing_genre_cols = [col for col in main_genres if col in song_features_df.columns]
song_features_matrix_map = {title: features for title, features in zip(song_features_df.index, song_features_df[existing_genre_cols].values)}

train_records = []
for u, i, r in trainset.all_ratings():
    train_records.append({'user_id': trainset.to_raw_uid(u), 'title': trainset.to_raw_iid(i), 'play_count': r})
train_df = pd.DataFrame(train_records)

def calculate_genre_similarity(predictions, train_df, k=10):
    user_est_true = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        user_est_true[uid].append((est, iid))
    all_users_avg_similarities = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        recommended_items = [iid for est, iid in user_ratings[:k]]
        user_train_data = train_df[train_df['user_id'] == uid]
        if not user_train_data.empty:
            merged_df = pd.merge(user_train_data, song_features_df[existing_genre_cols], left_on='title', right_index=True, how='inner')
            play_counts = merged_df['play_count']
            if play_counts.sum() > 0:
                user_profile_vector = (merged_df[existing_genre_cols].mul(play_counts, axis=0).sum() / play_counts.sum()).values.reshape(1, -1)
                current_user_similarities = []
                for song_title in recommended_items:
                    if song_title in song_features_matrix_map:
                        song_vector = song_features_matrix_map[song_title].reshape(1, -1)
                        similarity = cosine_similarity(user_profile_vector, song_vector)[0][0]
                        current_user_similarities.append(similarity)
                if current_user_similarities:
                    all_users_avg_similarities.append(np.mean(current_user_similarities))
    return np.mean(all_users_avg_similarities) if all_users_avg_similarities else 0
# -----------------


# --- 6. 💥💥💥 모델 성능 종합 평가 (정량 지표) 💥💥💥 ---
predictions = algo_for_evaluation.test(testset)
predictions_log = algo_for_evaluation_log.test(testset_log)

print("\n" + "="*60)
print("▶ 최종 모델 성능 평가 결과 (SVD: Log Transformation 효과 비교)")
print("="*60)

# 6-a. 예측 정확도 지표 (RMSE, MAE)
print("\n[1. 예측 정확도 지표 (without log)]\n")
accuracy.rmse(predictions, verbose=True)
accuracy.mae(predictions, verbose=True)
print("\n[1. 예측 정확도 지표 (with log)]\n")
accuracy.rmse(predictions_log, verbose=True)
accuracy.mae(predictions_log, verbose=True)

# 6-b. Top-N 추천 성능 지표 (Precision@k, Recall@k)
print(f"\n[2. Top-{K} 추천 성능 지표 (정답률)]\n")
precision, recall = precision_recall_at_k_implicit(predictions, k=K)
precision_log, recall_log = precision_recall_at_k_implicit(predictions_log, k=K)
print(f"Precision@{K} : Without log : {precision:.4f}, With log :{precision_log:.4f}")
print(f"Recall@{K}    : Without log : {recall:.4f}, With log : {recall_log:.4f}")
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
f1_score_log = 2 * (precision_log * recall_log) / (precision_log + recall_log) if (precision_log + recall_log) != 0 else 0
print(f"F1-Score@{K}  : Without log : {f1_score:.4f}, With log : {f1_score_log:.4f}")

# 6-c. 콘텐츠 정렬 성능 (장르 유사도)
print("\n" + "-"*60)
print(f"\n[3. Top-{K} 콘텐츠 정렬 성능 (취향 일치도)]\n")
avg_similarity = calculate_genre_similarity(predictions, train_df, k=K)
avg_similarity_log = calculate_genre_similarity(predictions_log, train_df, k=K)
print(f"Average Genre Similarity: Without log : {avg_similarity:.4f}, With log : {avg_similarity_log:.4f}")
print("\n(설명: 추천된 노래들이 사용자의 기존 청취 장르와 평균적으로 얼마나 유사한지를 나타냅니다.)")
print("="*60)


# --- 7. 💥💥💥 사례 연구 (Case Study) 결과 출력 💥💥💥 ---
print("\n" + "="*70)
print(f"▶ 사례 연구: SVD (with log) 모델의 랜덤 사용자 {K}개 추천 분석")
print("="*70)

user_predictions = defaultdict(list)
for uid, iid, _, est, _ in predictions_log:
    user_predictions[uid].append((iid, est))

num_samples = min(5, len(user_predictions))
random_users = random.sample(list(user_predictions.keys()), num_samples)

for i, user_id in enumerate(random_users):
    user_recs = sorted(user_predictions[user_id], key=lambda x: x[1], reverse=True)
    recommendations = [iid for iid, est in user_recs[:K]]
    
    print(f"\n--- [Case {i+1}] User: {user_id} ---\n")
    
    user_train_data = train_df[train_df['user_id'] == user_id]
    if not user_train_data.empty:
        merged_df = pd.merge(user_train_data, song_features_df[existing_genre_cols], left_on='title', right_index=True, how='inner')
        play_counts = merged_df['play_count']
        if play_counts.sum() > 0:
            user_profile_vector = (merged_df[existing_genre_cols].mul(play_counts, axis=0).sum() / play_counts.sum()).values
            user_top_genres = sorted(zip(existing_genre_cols, user_profile_vector), key=lambda x: x[1], reverse=True)
            
            print("  [사용자 주요 취향 (Top 5 Genres)]")
            for genre, score in user_top_genres[:5]:
                if score > 0.01:
                    print(f"    - {genre:<25} ({score:.2%})")
            
    print(f"\n  [SVD 추천 목록 (Top {K} Songs & Genres)]")
    if not recommendations:
        print("    (추천된 노래가 없습니다)")
    else:
        for song_title in recommendations:
            print(f"    - {song_title}")
            if song_title in song_features_matrix_map:
                song_vector = song_features_matrix_map[song_title]
                song_top_genres = sorted(zip(existing_genre_cols, song_vector), key=lambda x: x[1], reverse=True)
                genre_str = ", ".join([f"{g} " for g, s in song_top_genres[:5] if s > 0.01])
                print(f"      └─ Genres: {genre_str if genre_str else 'N/A'}")

print("\n" + "="*70)