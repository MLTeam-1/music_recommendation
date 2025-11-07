import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Surprise 라이브러리
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# --- 1. 설정 ---
FINAL_MERGED_CSV = '../data/final-lastfm-data.csv' # 최적화된 데이터 파일
K = 2  # Top-N 추천에서 사용할 N값 (보통 10을 많이 사용)

# --- 2. 데이터 로딩 ---
try:
    print(f"'{FINAL_MERGED_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(FINAL_MERGED_CSV)
    print("-> 로딩 완료.")
except FileNotFoundError:
    print(f"오류: '{FINAL_MERGED_CSV}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# Precision, Recall, F1-score 저장용 리스트
Eth = 20
precisions = []
recalls = []
f1_scores = []
rows = []

def precision_recall_at_k_implicit(predictions, k=10, est_threshold=1.05):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        n_rel = sum((true_r > 0) for (_, true_r) in user_ratings)
        n_rel_and_rec_k = sum(((true_r > 0)) for (_, true_r) in user_ratings[:k])
        
        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return sum(p for p in precisions.values()) / len(precisions), \
           sum(r for r in recalls.values()) / len(recalls)

user_counts = df['user_id'].value_counts()
active_users = user_counts[user_counts >= Eth].index
filtered_df = df[df['user_id'].isin(active_users)]  
filtered_df = filtered_df.copy()  # 안전하게 복사본 생성한 뒤
filtered_df['rating'] = np.log1p(filtered_df['play_count'])


reader = Reader(rating_scale=(filtered_df['rating'].min(), filtered_df['rating'].max()))
data = Dataset.load_from_df(filtered_df[['user_id', 'title', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=41)

cnt=Eth
algo = SVD(n_factors=100, n_epochs=20, random_state=42)
algo.fit(trainset)

predictions = algo.test(testset)

rating_threshold = filtered_df['rating'].mean()

precision, recall = precision_recall_at_k_implicit(predictions, k=K, est_threshold=rating_threshold)

f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

test_user_counts = filtered_df['user_id'].value_counts()


# 평균 '정답' 수 계산
average_items_in_testset = test_user_counts.mean()

print(f"\nTest Set에서 사용자별 평균 청취 기록(정답)의 수: {average_items_in_testset:.2f} 개")

k = int(average_items_in_testset)
k_values = list(range(max(k-10, 1), k+10))
for k in k_values:
    precision, recall = precision_recall_at_k_implicit(predictions, k=k)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)


# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(10, 6))

color_metrics = 'tab:blue'

# 주 y축: 평가 지표
ax1.set_xlabel('Minimum Listen Count Threshold (Eth)')
ax1.set_ylabel('Evaluation Metrics', color=color_metrics)
ax1.plot(k_values, precisions, label='Precision@', marker='o', color='blue')
ax1.plot(k_values, recalls, label='Recall@', marker='x', color='orange')
ax1.plot(k_values, f1_scores, label='F1-Score@', marker='s', color='red')
ax1.tick_params(axis='y', labelcolor=color_metrics)
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_xlim(min(k_values), max(k_values))

plt.title(f'LastFM K value compresion')
fig.tight_layout()
plt.savefig('LastFM K value compresion in fixed Minimum_listen_count.png')
plt.show()