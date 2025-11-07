# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] 콘텐츠 기반 필터링(Content-Based Filtering) 모델 구현 및 성능 평가
# [End-to-End 단계] 5. 모델링(Modeling) 및 6. 평가(Evaluation)
#
# [설명]
# 이 스크립트는 콘텐츠 기반 필터링의 핵심 로직을 구현합니다.
# 각 사용자가 과거에 청취한 노래들의 '장르' 정보를 바탕으로, 해당 사용자의 고유한
# '음악 취향 프로필 벡터(User Profile Vector)'를 생성합니다.
#
# 그 후, 학습에 사용되지 않은 Test Set의 노래들에 대해, 이 프로필 벡터와의
# '코사인 유사도(Cosine Similarity)'를 계산하여 예측 선호도를 구합니다.
#
# 최종적으로, 이 예측 선호도를 기반으로 Top-N 추천 성능 지표(Precision@k, Recall@k)를
# 계산하여 모델의 성능을 정량적으로 평가하는 것을 목표로 합니다.
# ======================================================================================

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 1. 설정 ---
# 이 섹션에서는 모델과 실험에 사용될 주요 파라미터들을 정의합니다.
FINAL_FILTERED_CSV = '../../data/final-lastfm-data.csv' # 전처리가 완료된 데이터 파일
# -----------------


# --- 2. 데이터 로딩 ---
# 이 섹션에서는 CSV 파일을 불러와 모델 학습에 적합한 형태로 가공합니다.
try:
    print(f"'{FINAL_FILTERED_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(FINAL_FILTERED_CSV)
    # SVD 모델과의 공정한 비교를 위해, 동일하게 로그 변환된 'play_count'를 'rating'으로 사용합니다.
    df['rating'] = np.log1p(df['play_count']) 
    print("-> 로딩 완료.")
except FileNotFoundError:
    print(f"오류: '{FINAL_FILTERED_CSV}' 파일을 찾을 수 없습니다.")
    exit()
# -----------------

    
# --- 3. 데이터 분리 (학습용 / 평가용) ---
# 이 섹션은 모델의 성능을 공정하게 평가하기 위해 데이터를 분리하는 매우 중요한 단계입니다.
# 데이터 유출(Data Leakage)을 방지하는 것이 핵심 목표입니다.

# [핵심] 안정적인 데이터 분할을 위한 사용자 필터링
# train_test_split의 'stratify' 옵션은 각 사용자의 데이터를 지정된 비율로 나눕니다.
# 만약 사용자의 기록이 너무 적으면(예: 1개), 8:2로 나눌 수 없어 오류가 발생합니다.
# 따라서, 분할에 필요한 최소 기록 수를 가진 사용자만 남겨 안정성을 확보합니다.
MIN_RECORDS_FOR_SPLIT = 15
user_counts = df['user_id'].value_counts()
users_with_enough_records = user_counts[user_counts >= MIN_RECORDS_FOR_SPLIT].index
df_filtered_for_split = df[df['user_id'].isin(users_with_enough_records)]

print(f"\n데이터 분할을 위해, 청취 기록이 {MIN_RECORDS_FOR_SPLIT}개 미만인 사용자들을 제외합니다.")
print(f"원본 데이터: {len(df)}개 -> 필터링 후 데이터: {len(df_filtered_for_split)}개")
print(f"분석 대상 사용자 수: {len(users_with_enough_records)}명")

print("\n데이터를 Train Set (80%)과 Test Set (20%)으로 분리합니다...")
# [핵심] 'stratify' 옵션 사용 이유:
# 이 옵션을 사용하면, 필터링된 모든 사용자가 Train Set과 Test Set에 8:2 비율로
# 공평하게 포함됩니다. 이를 통해 특정 사용자가 Test Set에 없어 평가가 불가능해지는
# 문제를 방지하고, 모든 사용자에 대해 모델을 평가할 수 있게 됩니다.
train_df, test_df = train_test_split(
    df_filtered_for_split, 
    test_size=0.2, 
    stratify=df_filtered_for_split['user_id'], 
    random_state=42 # 재현성을 위해 random_state 고정
)
print(f"-> Train Set: {len(train_df)}개, Test Set: {len(test_df)}개")
# -----------------


# --- 4. 콘텐츠 기반 필터링 예측 수행 ---
# 이 섹션에서는 Test Set의 각 (사용자, 아이템) 쌍에 대한 선호도를 예측합니다.

print("\n콘텐츠 기반 모델로 Test Set의 평점을 예측합니다...")
# 노래의 콘텐츠 정보(장르 벡터)를 빠르게 조회할 수 있도록 딕셔너리 형태로 구축
# 참고: 노래의 장르 정보 자체는 사용자의 청취 기록과 무관한 '사전 정보'이므로,
#       전체 데이터프레임(df)을 사용하여 구축해도 데이터 유출에 해당하지 않습니다.
song_features_df = df.drop_duplicates(subset='title').set_index('title')
main_genres = ['Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 'Folk & Country', 'Reggae', 'Other']
existing_genre_cols = [col for col in main_genres if col in song_features_df.columns]
song_features_matrix_map = {title: features for title, features in zip(song_features_df.index, song_features_df[existing_genre_cols].values)}

predictions = []
# tqdm: 처리 진행 상황을 시각적으로 보여주는 라이브러리
for user_id, user_test_data in tqdm(test_df.groupby('user_id'), desc="Predicting for users"):
    # [Step 1] 사용자 프로필 생성 (오직 Train Set 데이터만 사용)
    user_train_data = train_df[train_df['user_id'] == user_id]
    if user_train_data.empty:
        continue # Stratify 옵션 덕분에 이 경우는 거의 발생하지 않음
        
    # 사용자가 들었던 노래들의 장르 벡터에 'play_count'를 가중치로 곱하여 합산 -> 가중 평균
    # 즉, 많이 들은 노래의 장르가 사용자의 취향에 더 큰 영향을 미치도록 합니다.
    user_genre_data = user_train_data[existing_genre_cols]
    play_counts = user_train_data['play_count']
    user_profile_vector = (user_genre_data.mul(play_counts, axis=0).sum() / play_counts.sum()).values.reshape(1, -1)
    
    # [Step 2] Test Set 아이템에 대한 선호도 예측
    for index, row in user_test_data.iterrows():
        title = row['title']
        true_rating = row['rating'] # 실제 정답 값 (평가에 사용)
        
        if title in song_features_matrix_map:
            song_vector = song_features_matrix_map[title].reshape(1, -1)
            # [핵심 로직] 사용자 프로필 벡터와 노래의 장르 벡터 간의 코사인 유사도를 계산
            # 이 유사도 점수(0~1)가 바로 모델이 예측한 '예상 선호도(est)'가 됩니다.
            estimated_rating = cosine_similarity(user_profile_vector, song_vector)[0][0]
            
            # 평가를 위해 예측 결과를 저장
            predictions.append({'uid': user_id, 'iid': title, 'r_ui': true_rating, 'est': estimated_rating})
# -----------------


# --- 5. 모델 성능 평가 ---
# 이 섹션에서는 저장된 예측 결과를 바탕으로 모델의 성능을 정량적으로 측정합니다.
print("\n" + "="*60)
print("▶ 최종 모델 성능 평가 결과 (Content-Based Filtering)")
print("="*60)

# 5-a. 예측 정확도 지표 (RMSE, MAE)
true_ratings = [p['r_ui'] for p in predictions]
estimated_ratings = [p['est'] for p in predictions]

# [중요] 코사인 유사도(0~1)와 실제 rating(로그 변환된 play_count)은 스케일이 다릅니다.
# 따라서, RMSE/MAE는 두 값의 차이를 보여줄 뿐, 모델의 '랭킹' 성능을 직접적으로
# 나타내지는 않으므로 '참고용'으로만 해석해야 합니다.
rmse = np.sqrt(mean_squared_error(true_ratings, estimated_ratings))
mae = mean_absolute_error(true_ratings, estimated_ratings)

print("\n[1. 예측 정확도 지표 (참고용)]\n")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# 5-b. Top-N 추천 성능 지표 (Precision@k, Recall@k)
# 이 지표들이야말로 추천 목록의 품질을 측정하는 핵심적인 성능 척도입니다.
def precision_recall_at_k_implicit(predictions, k=10):
    user_est_true = defaultdict(list)
    for p in predictions:
        user_est_true[p['uid']].append((p['est'], p['r_ui']))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r > 0) for (_, true_r) in user_ratings)
        n_rel_and_rec_k = sum(((true_r > 0)) for (_, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return sum(p for p in precisions.values()) / len(precisions), \
           sum(r for r in recalls.values()) / len(recalls)
           
test_user_counts = test_df['user_id'].value_counts()
# K값을 임의로 정하는 대신, Test Set의 사용자별 평균 아이템 수를 기준으로 설정
# 이는 데이터에 기반한 합리적인 K값 설정 방식입니다.
average_items_in_testset = test_user_counts.mean()
print(f"\nTest Set에서 사용자별 평균 청취 기록(정답)의 수: {average_items_in_testset:.2f} 개")

k = int(round(average_items_in_testset))

# 데이터 기반으로 설정된 k값 주변의 여러 k에 대해 성능을 종합적으로 확인
for k_val in range(max(k-5, 1), k+5):
    precision, recall = precision_recall_at_k_implicit(predictions, k=k_val)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print(f"\n[2. Top-{k_val} 추천 성능 지표]\n")
    print(f"Precision@{k_val} (Implicit): {precision:.4f}")
    print(f"Recall@{k_val} (Implicit)   : {recall:.4f}")
    print(f"F1-Score@{k_val} : {f1_score:.4f}")
# ======================================================================================