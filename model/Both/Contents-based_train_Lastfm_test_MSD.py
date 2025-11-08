# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] 교차 데이터셋 환경에서 콘텐츠 기반 모델의 성능 평가 및 사례 연구
# [End-to-End 단계] 7. 모델 평가(Evaluation) 및 8. 사례 연구(Case Study)
#
# [설명]
# 이 스크립트는 "Train on Last.fm, Test on MSD"라는 교차 데이터셋 실험을 수행합니다.
# 1. Last.fm 데이터('품질' 데이터)를 사용하여 각 사용자의 '취향 프로필'을 생성합니다. (학습)
# 2. 이 프로필을 기반으로 MSD 데이터('양' 데이터)에 있는 노래들 중에서 Top-K개를 추천합니다. (예측)
# 3. '취향 일치도'를 측정하여 추천 목록의 장르 분포가 사용자의 원래 취향과 얼마나 유사한지 평가합니다.
# 4. [추가] 실제 추천 사례를 랜덤으로 샘플링하여, 모델의 추천 행동을 질적으로 분석합니다.
# ======================================================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random # 💥💥💥 랜덤 샘플링을 위해 임포트 💥💥💥

# --- 1. 설정 ---
# 💥💥💥 파일 경로 이름 변경 (Train/Test 혼동 방지) 💥💥💥
PROFILE_DATA_CSV = '../../data/final-lastfm-data.csv' # 사용자 프로필 생성용 (Last.fm)
RECOMMEND_POOL_CSV = '../../data/final-msd-data.csv'  # 추천 대상 노래 목록 (MSD)

MIN_USER_RECORDS = 10
K_FOR_RANKING = 10

# --- 2. 데이터 로딩 ---
try:
    print(f"'{PROFILE_DATA_CSV}' (프로필 생성용) 파일을 읽는 중...")
    df_profile = pd.read_csv(PROFILE_DATA_CSV)
    print(f"'{RECOMMEND_POOL_CSV}' (추천 대상용) 파일을 읽는 중...")
    df_pool = pd.read_csv(RECOMMEND_POOL_CSV)
    print("-> 로딩 완료.")
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. {e}")
    exit()

# --- 3. 데이터 준비 ---
print("\n--- 2. 데이터 준비 시작 ---")

# 3-1. 활성 사용자 필터링 (프로필 생성용 데이터 기준)
print(f"프로필 데이터에서 청취 기록이 {MIN_USER_RECORDS}개 이상인 '활성 사용자'를 찾습니다...")
user_counts = df_profile['user_id'].value_counts()
active_users = user_counts[user_counts >= MIN_USER_RECORDS].index.tolist()
print(f"-> 총 {len(active_users)}명의 활성 사용자를 대상으로 평가를 진행합니다.")

# 3-2. 콘텐츠 정보(장르 벡터) 준비
combined_df = pd.concat([df_profile, df_pool])
main_genres = ['Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 'Folk & Country', 'Reggae', 'Other', 'Rock']
existing_genre_cols = [col for col in main_genres if col in combined_df.columns]

# 추천 대상이 될 MSD 노래들의 장르 벡터 미리 준비
msd_song_features = df_pool.drop_duplicates(subset='title')[['title'] + existing_genre_cols].set_index('title')
print("-> 추천 대상 노래(MSD)의 장르 정보 준비 완료.")
print("--- 2. 데이터 준비 완료 ---\n")


# --- 4. 핵심 로직: '취향 일치도' 평가 및 사례 수집 ---
print("--- 3. 콘텐츠 기반 추천 생성 및 '취향 일치도' 평가 시작 ---")

profile_similarities = []
case_study_results = {} # 💥💥💥 사례 연구를 위한 딕셔너리 추가 💥💥💥

for user_id in tqdm(active_users, desc="각 사용자에 대한 추천 및 '취향 일치도' 평가 중"):
    
    # 4-1. 사용자 취향 프로필 벡터 생성 (Last.fm 데이터 사용)
    user_listen_history = df_profile[df_profile['user_id'] == user_id]
    if user_listen_history.empty:
        continue
    play_counts = user_listen_history['play_count']
    user_profile_vector = np.average(user_listen_history[existing_genre_cols], axis=0, weights=play_counts).reshape(1, -1)
    
    # 4-2. Top-K 추천 목록 생성 (MSD 노래 중에서)
    listened_songs_total = set(df_profile[df_profile['user_id'] == user_id]['title'].unique()).union(
                           set(df_pool[df_pool['user_id'] == user_id]['title'].unique()))
    
    recommendation_candidates = msd_song_features[~msd_song_features.index.isin(listened_songs_total)]
    if recommendation_candidates.empty:
        continue
        
    similarity_scores = cosine_similarity(user_profile_vector, recommendation_candidates.values)[0]
    song_scores = list(zip(recommendation_candidates.index, similarity_scores))
    song_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_recs = song_scores[:K_FOR_RANKING]
    recommended_titles = [title for title, score in top_n_recs]
    
    if not recommended_titles:
        continue
    
    # 4-3. '취향 일치도' 계산
    recommended_songs_features = msd_song_features.loc[recommended_titles]
    avg_recommendation_vector = recommended_songs_features.mean().values.reshape(1, -1)
    profile_similarity_score = cosine_similarity(user_profile_vector, avg_recommendation_vector)[0][0]
    profile_similarities.append(profile_similarity_score)

    # 💥💥💥 [사례 연구] 현재 사용자의 분석 결과를 딕셔너리에 저장 💥💥💥
    case_study_results[user_id] = {
        'profile_vector': user_profile_vector.flatten(),
        'recommendations': recommended_titles
    }

print("--- 3. 평가 완료 ---\n")


# --- 5. 최종 성능 지표 요약 ---
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


# --- 6. 💥💥💥 사례 연구 (Case Study) 결과 출력 💥💥💥 ---
print("\n" + "="*70)
print("▶ 사례 연구: 랜덤 사용자 5명에 대한 심층 분석 (Train on Last.fm, Rec on MSD)")
print("="*70)

num_samples = min(5, len(case_study_results))
random_users = random.sample(list(case_study_results.keys()), num_samples)

for i, user_id in enumerate(random_users):
    user_data = case_study_results[user_id]
    profile_vector = user_data['profile_vector']
    recommendations = user_data['recommendations']
    
    print(f"\n--- [Case {i+1}] User: {user_id} ---\n")
    
    # 1. 사용자의 Top 5 선호 장르 출력 (Last.fm 기반)
    user_top_genres = sorted(zip(existing_genre_cols, profile_vector), key=lambda x: x[1], reverse=True)
    print("  [사용자 주요 취향 (Top 5 Genres from Last.fm)]")
    for genre, score in user_top_genres[:5]:
        if score > 0.01:
            print(f"    - {genre:<25} ({score:.2%})")
            
    # 2. 추천된 노래 목록과 각 노래의 장르 출력 (MSD 노래들)
    print(f"\n  [Content-Based 추천 목록 (Top {K_FOR_RANKING} Songs from MSD & Genres)]")
    if not recommendations:
        print("    (추천된 노래가 없습니다)")
    else:
        for song_title in recommendations:
            print(f"    - {song_title}")
            if song_title in msd_song_features.index:
                song_vector = msd_song_features.loc[song_title].values
                song_top_genres = sorted(zip(existing_genre_cols, song_vector), key=lambda x: x[1], reverse=True)
                genre_str = ", ".join([f"{g} " for g, s in song_top_genres[:5] if s > 0.01])
                print(f"      └─ Genres: {genre_str if genre_str else 'N/A'}")

print("\n" + "="*70)