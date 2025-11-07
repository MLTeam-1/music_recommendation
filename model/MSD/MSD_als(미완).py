import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import os
from tqdm import tqdm

# --- 1. 설정 ---
DATA_PATH = '../../data/final-msd-data.csv'
K_FOR_RANKING = 10
Eth = 1 # 최소 청취 횟수 임계값
factors = 50
regularization = 0.01
iterations = 20
random_state = 42

# --- 2. 데이터 준비 ---
print("--- 1. 데이터 준비 시작 ---")
try:
    df_raw = pd.read_csv(DATA_PATH)
    print(f"'{DATA_PATH}' 파일 로드 성공. (원본 행 수: {len(df_raw):,})")
except FileNotFoundError:
    print(f"오류: '{DATA_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 사용자 필터링
print(f"\n청취 기록이 {Eth}개 이상인 활성 사용자만 필터링합니다...")
user_counts = df_raw['user_id'].value_counts()
active_users = user_counts[user_counts >= Eth].index
df = df_raw[df_raw['user_id'].isin(active_users)].copy()
print(f"-> 필터링 후 남은 행 수: {len(df):,}")

print("\n데이터를 implicit 라이브러리 형식으로 변환 중...")

# ID와 정수 인덱스 명시적 매핑
unique_users = df['user_id'].unique()
unique_items = df['title'].unique()
user_to_idx = {user: i for i, user in enumerate(unique_users)}
item_to_idx = {item: i for i, item in enumerate(unique_items)}
idx_to_user = {i: user for user, i in user_to_idx.items()}
idx_to_item = {i: item for item, i in item_to_idx.items()}

user_indices = df['user_id'].map(user_to_idx)
item_indices = df['title'].map(item_to_idx)
num_users = len(unique_users)
num_items = len(unique_items)
print(f"-> 최종 고유 사용자 수: {num_users}, 최종 고유 아이템 수: {num_items}")

# --- 3. 수동으로 Train / Test 데이터 분할 ---
print("\n--- 2. 데이터를 Train / Test 세트로 수동 분할 ---")
train_list = []
test_list = []
# .groupby('user_id')를 사용하여 각 사용자별로 데이터를 80:20으로 나눕니다.
for _, group in tqdm(df.groupby('user_id'), desc="Splitting data by user"):
    # 각 사용자의 데이터가 2개 미만이면 모두 train으로 보냄 (평가 불가능)
    if len(group) < 2:
        train_list.append(group)
        continue
    
    # 각 사용자의 데이터를 랜덤하게 섞은 후, 80%를 train으로 선택
    frac = 0.8
    train_sample = group.sample(frac=frac, random_state=random_state)
    test_sample = group.drop(train_sample.index)
    
    train_list.append(train_sample)
    test_list.append(test_sample)

# 분할된 데이터들을 다시 하나의 DataFrame으로 합침
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

# Train 데이터로만 학습용 희소 행렬 생성
train_user_indices = train_df['user_id'].map(user_to_idx)
train_item_indices = train_df['title'].map(item_to_idx)
train_matrix = csr_matrix((train_df['play_count'].astype(float),
                           (train_user_indices, train_item_indices)),
                          shape=(num_users, num_items))
train_item_user = train_matrix.T.tocsr()
print("-> 데이터 분할 완료.")
print(f"Train Set 크기: {len(train_df)} 상호작용, Test Set 크기: {len(test_df)} 상호작용")


# --- 4. ALS 모델 학습 ---
print("\n--- 3. ALS 모델 학습 시작 (Train Set 사용) ---")
model = AlternatingLeastSquares(factors=factors,
                                regularization=regularization,
                                iterations=iterations,
                                random_state=random_state)
model.fit(train_item_user, show_progress=True)
print("--- 3. 모델 학습 완료 ---\n")


# --- 5. 💥💥💥 전면 수정된 Top-N 성능 평가 로직 (recommend_all 사용) 💥💥💥 ---
print("--- 4. 모델 성능 평가 시작 (recommend_all 사용) ---")

# 5.1. 정답지(Test Set) 및 학습 기록(Train Set) 준비
true_relevants = test_df.groupby('user_id')['title'].apply(set).to_dict()
train_relevants_indices = train_df.groupby('user_id')['title'].apply(lambda x: {item_to_idx[i] for i in x}).to_dict()

# 5.2. 평가 대상 사용자 목록
test_users_indices = [user_to_idx[user] for user in true_relevants.keys() if user in user_to_idx]

# 각 사용자의 평가 결과를 저장할 리스트
precisions = []
recalls = []

# 5.3. 각 사용자에 대해 추천 생성 및 평가
for user_idx in tqdm(test_users_indices, desc="Evaluating"):
    user_id_str = idx_to_user[user_idx]

    # [핵심 수정 1] recommend_all 함수 사용
    # 이 함수는 user_items 없이, 모든 아이템에 대한 점수를 반환합니다.
    scores = model.recommend_all(user_items=train_matrix[user_idx])
    
    # [핵심 수정 2] 점수 배열을 (아이템 인덱스, 점수) 쌍으로 변환 후 정렬
    all_recommendations = list(enumerate(scores))
    all_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # [핵심 수정 3] 수동으로 '이미 본 아이템' 필터링 및 Top-K 선정
    already_liked_indices = train_relevants_indices.get(user_id_str, set())
    
    filtered_recs_indices = []
    for item_idx, score in all_recommendations:
        if item_idx not in already_liked_indices:
            filtered_recs_indices.append(item_idx)
        if len(filtered_recs_indices) >= K_FOR_RANKING:
            break
            
    recommended_items = {idx_to_item[idx] for idx in filtered_recs_indices}
    
    # 이 사용자의 실제 정답 아이템 목록 (Test Set)
    ground_truth_items = true_relevants.get(user_id_str, set())
    
    if not ground_truth_items:
        continue
        
    # 추천된 것과 실제 정답이 겹치는 아이템 수
    intersection = recommended_items.intersection(ground_truth_items)
    
    # Precision, Recall 계산
    precisions.append(len(intersection) / K_FOR_RANKING)
    recalls.append(len(intersection) / len(ground_truth_items))

# 5.4. 최종 평균 점수 계산
avg_precision = np.mean(precisions) if precisions else 0
avg_recall = np.mean(recalls) if recalls else 0
f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) != 0 else 0
print("\n--- 4. 평가 완료 ---\n")

# --- 6. 최종 결과 출력 ---
print("="*60)
print(f"▶ ALS 모델 성능 평가 요약 (MSD, Filtered)")
print("="*60)
print(f"\n[Top-{K_FOR_RANKING} 추천 성능 지표]\n")
print(f"Precision@{K_FOR_RANKING}: {avg_precision:.4f}")
print(f"Recall@{K_FOR_RANKING}   : {avg_recall:.4f}")
print(f"F1-Score@{K_FOR_RANKING}  : {f1_score:.4f}")
print("="*60)