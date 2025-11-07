import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import os

# --- 1. 설정 (이전과 동일) ---
DATA_PATH = '../../data/final-lastfm-data.csv'
TARGET_USER = None
N_RECOMMENDATIONS = 10
factors = 50
regularization = 0.01
iterations = 20
random_state = 42

# --- 2. 💥💥💥 전면 수정된 데이터 준비 로직 💥💥💥 ---
print("--- 1. 데이터 준비 시작 ---")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"'{DATA_PATH}' 파일 로드 성공.")
except FileNotFoundError:
    print(f"오류: '{DATA_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

print("데이터를 implicit 라이브러리 형식으로 변환 중...")

# [핵심 수정 1] ID와 정수 인덱스를 직접, 명시적으로 매핑
# 1-1. 고유한 사용자/아이템 목록 생성
unique_users = df['user_id'].unique()
unique_items = df['title'].unique()

# 1-2. 문자열 ID -> 정수 인덱스 매핑 딕셔너리 생성
user_to_idx = {user: i for i, user in enumerate(unique_users)}
item_to_idx = {item: i for i, item in enumerate(unique_items)}

# 1-3. 정수 인덱스 -> 문자열 ID 매핑 딕셔너리 생성 (나중에 결과를 변환하기 위함)
idx_to_user = {i: user for user, i in user_to_idx.items()}
idx_to_item = {i: item for item, i in item_to_idx.items()}

# 1-4. 원본 데이터프레임의 문자열 ID를 우리가 만든 정수 인덱스로 변환
user_indices = df['user_id'].map(user_to_idx)
item_indices = df['title'].map(item_to_idx)

# [핵심 수정 2] 직접 만든 인덱스를 사용하여 희소 행렬 생성
# 이제 행렬의 크기와 인덱스의 범위가 완벽하게 일치하는 것이 보장됩니다.
num_users = len(unique_users)
num_items = len(unique_items)

interaction_matrix = csr_matrix((df['play_count'].astype(float),
                                 (user_indices,
                                  item_indices)),
                                shape=(num_users, num_items))

# 아이템-사용자 행렬로 변환 (학습용)
item_user_matrix = interaction_matrix.T.tocsr()

print(f"-> 고유 사용자 수: {num_users}, 고유 아이템 수: {num_items}")
print("--- 1. 데이터 준비 완료 ---\n")


# --- 3. ALS 모델 학습 (이전과 동일) ---
print("--- 2. ALS 모델 학습 시작 ---")
model = AlternatingLeastSquares(factors=factors,
                                regularization=regularization,
                                iterations=iterations,
                                random_state=random_state)
model.fit(item_user_matrix, show_progress=True)
print("--- 2. ALS 모델 학습 완료 ---\n")


# --- 4. 특정 사용자를 위한 추천 생성 (타겟 사용자 찾는 방식 수정) ---
print("--- 3. 특정 사용자를 위한 추천 생성 시작 ---")
if TARGET_USER is None:
    TARGET_USER = df['user_id'].iloc[0]
print(f"타겟 사용자: {TARGET_USER}")

try:
    # 이제 우리가 만든 user_to_idx 딕셔너리를 사용
    target_user_idx = user_to_idx[TARGET_USER]
except KeyError:
    print(f"오류: '{TARGET_USER}'는 데이터에 없는 사용자입니다.")
    exit()

# model.recommend() 함수 호출
recommended_indices, scores = model.recommend(
    userid=target_user_idx,
    user_items=interaction_matrix[target_user_idx],
    N=N_RECOMMENDATIONS,
    filter_already_liked_items=True
)

# [핵심 수정 3] 추천된 아이템의 정수 인덱스를 실제 노래 제목으로 변환
top_n_recommendations = [(idx_to_item[idx], score) for idx, score in zip(recommended_indices, scores)]
print("--- 3. 추천 생성 완료 ---\n")


# --- 5. 최종 추천 결과 출력 (이전과 동일) ---
print("="*60)
print(f"'{TARGET_USER}'님을 위한 Top {N_RECOMMENDATIONS} 음악 추천 (ALS)")
print("="*60)
for i, (song_title, score) in enumerate(top_n_recommendations):
    print(f"{i+1:2d}. {song_title:<40} (추천 점수: {score:.4f})")
print("="*60)