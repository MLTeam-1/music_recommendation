# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] 콘텐츠 기반 필터링 모델의 성능을 '취향 일치도' 지표로 평가
# [End-to-End 단계] 7. 모델 평가(Evaluation)
#
# [설명]
# 이 스크립트는 "Train on Last.fm, Test on MSD"라는 교차 데이터셋 실험을 수행하고,
# 그 성능을 객관적인 수치로 평가하는 것을 목표로 합니다.
# 1. Last.fm 데이터('품질' 데이터)를 사용하여 각 사용자의 '취향 프로필'을 생성합니다. (학습)
# 2. 이 프로필을 기반으로 MSD 데이터('양' 데이터)에 있는 노래들 중에서 Top-K개를 추천합니다. (예측)
# 3. 전통적인 정확도 지표(Precision/Recall)의 한계를 보완하기 위해, '취향 일치도'라는
#    보조 지표를 사용하여 "추천된 노래 목록의 장르 분포가 사용자의 원래 취향과 얼마나 유사한지"를
#    코사인 유사도로 측정합니다. (평가)
# 최종적으로, 모든 사용자에 대한 평균 '취향 일치도' 점수를 출력하여 모델의 전반적인 성능을 요약합니다.
# ======================================================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- 1. 설정 ---
# 사용자 프로필 생성을 위한 학습(TEST_CSV) 데이터
TEST_CSV = '../../data/final-lastfm-data.csv' 
# 추천 대상 노래 목록이 포함된 테스트(TRAIN_CSV) 데이터
TRAIN_CSV = '../../data/final-msd-data.csv' 

# 프로필을 생성할 사용자를 필터링하기 위한 최소 청취 기록 수
MIN_USER_RECORDS = 10
# 각 사용자에게 추천할 상위 노래의 수
K_FOR_RANKING = 10

# --- 2. 데이터 로딩 ---
try:
    print(f"'{TRAIN_CSV}' (학습용) 파일을 읽는 중...")
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"'{TEST_CSV}' (평가용) 파일을 읽는 중...")
    df_test = pd.read_csv(TEST_CSV)
    print("-> 로딩 완료.")
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. {e}")
    exit()

# --- 3. 데이터 준비 ---
# 이 블록은 모델링과 평가에 필요한 데이터 구조를 미리 준비하는 단계입니다.
print("\n--- 2. 데이터 준비 시작 ---")

# 3-1. 활성 사용자 필터링
# 학습 데이터에서 청취 기록이 충분한 사용자만 평가 대상으로 삼습니다.
print(f"학습 데이터에서 청취 기록이 {MIN_USER_RECORDS}개 이상인 '활성 사용자'를 찾습니다...")
user_counts = df_train['user_id'].value_counts()
active_users = user_counts[user_counts >= MIN_USER_RECORDS].index.tolist()
print(f"-> 총 {len(active_users)}명의 활성 사용자를 대상으로 평가를 진행합니다.")

# 3-2. 콘텐츠 정보(장르 벡터) 준비
# 두 데이터셋을 합쳐 전체 노래의 장르 정보를 통합적으로 관리합니다.
combined_df = pd.concat([df_train, df_test])
main_genres = ['Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 'Folk & Country', 'Reggae', 'Other', 'Rock']
existing_genre_cols = [col for col in main_genres if col in combined_df.columns]

# 추천 대상이 될 Test Set(MSD) 노래들의 장르 벡터를 미리 준비하여 검색 속도를 높입니다.
msd_song_features = df_test.drop_duplicates(subset='title')[['title'] + existing_genre_cols].set_index('title')
print("-> 추천 대상 노래(Test Set)의 장르 정보 준비 완료.")
print("--- 2. 데이터 준비 완료 ---\n")


# --- 4. 핵심 로직: '취향 일치도' 평가 ---
print("--- 3. 콘텐츠 기반 추천 생성 및 '취향 일치도' 평가 시작 ---")

# 각 사용자의 '취향 일치도' 점수를 저장할 리스트
profile_similarities = []

# tqdm을 사용하여 전체 진행 상황을 시각적으로 보여줍니다.
for user_id in tqdm(active_users, desc="각 사용자에 대한 추천 및 '취향 일치도' 평가 중"):
    
    # 4-1. 사용자 취향 프로필 벡터 생성 (Train Set 사용)
    user_listen_history = df_train[df_train['user_id'] == user_id]
    if user_listen_history.empty:
        continue
    play_counts = user_listen_history['play_count']
    user_profile_vector = np.average(user_listen_history[existing_genre_cols], axis=0, weights=play_counts).reshape(1, -1)
    
    # 4-2. Top-K 추천 목록 생성 (Test Set의 노래 중에서)
    # 추천에서 제외할, 사용자가 이미 들어본 노래 목록을 만듭니다.
    listened_songs_total = set(df_train[df_train['user_id'] == user_id]['title'].unique()).union(
                           set(df_test[df_test['user_id'] == user_id]['title'].unique()))
    
    # 아직 듣지 않은 노래만 추천 후보군으로 필터링합니다.
    recommendation_candidates = msd_song_features[~msd_song_features.index.isin(listened_songs_total)]
    if recommendation_candidates.empty:
        continue
        
    # 사용자 프로필과 후보군 노래들 간의 코사인 유사도를 계산하고, 상위 K개를 선택합니다.
    similarity_scores = cosine_similarity(user_profile_vector, recommendation_candidates.values)[0]
    song_scores = list(zip(recommendation_candidates.index, similarity_scores))
    song_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_recs = song_scores[:K_FOR_RANKING]
    recommended_titles = [title for title, score in top_n_recs]
    
    if not recommended_titles:
        continue
    
    # [핵심 평가 로직]
    # 4-3. 추천 목록의 '평균 장르 벡터' 생성
    # 추천된 K개 노래들의 평균적인 장르 분포를 계산합니다.
    recommended_songs_features = msd_song_features.loc[recommended_titles]
    avg_recommendation_vector = recommended_songs_features.mean().values.reshape(1, -1)
    
    # 4-4. '취향 일치도' (코사인 유사도) 계산
    # (사용자 프로필 벡터)와 (추천 목록의 평균 장르 벡터) 간의 유사도를 측정합니다.
    profile_similarity_score = cosine_similarity(user_profile_vector, avg_recommendation_vector)[0][0]
    profile_similarities.append(profile_similarity_score)

print("--- 3. 평가 완료 ---\n")

# --- 5. 최종 성능 지표 요약 ---
# 모든 사용자에 대해 계산된 '취향 일치도' 점수들의 평균을 구합니다.
avg_profile_similarity = np.mean(profile_similarities) if profile_similarities else 0

print("="*60)
print(f"▶ 콘텐츠 기반 모델 '취향 일치도' 평가 요약")
print(f"(Train on Last.fm, Test on MSD)")
print("="*60)
print(f"평가 대상 사용자 수: {len(profile_similarities)}명\n")

print(f"[추천 품질 지표 (Top-{K_FOR_RANKING})]\n")
print(f"평균 '취향 일치도': {avg_profile_similarity:.4f}")
print(" (사용자 프로필 vs 추천 목록 간 코사인 유사도 평균)")
print("="*60)
print("\n(해석: 이 점수가 1에 가까울수록, 추천된 노래 목록의 장르 '색깔'이")
print("사용자의 전반적인 장르 선호도와 일치한다는 의미입니다.)")