import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Surprise 라이브러리
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# --- 1. 설정 ---
FINAL_MERGED_CSV = '../data/final-msd-data.csv' # 최적화된 데이터 파일
K = 5  # Top-N 추천에서 사용할 N값 (보통 10을 많이 사용)

# --- 2. 데이터 로딩 ---
try:
    print(f"'{FINAL_MERGED_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(FINAL_MERGED_CSV)
    print("-> 로딩 완료.")
except FileNotFoundError:
    print(f"오류: '{FINAL_MERGED_CSV}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# Precision, Recall, F1-score 저장용 리스트
eth_values = list(range(5, 51))
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
cnt=0
for Eth in eth_values:
    user_counts = df['user_id'].value_counts()
    active_users = user_counts[user_counts >= Eth].index
    filtered_df = df[df['user_id'].isin(active_users)]  
    filtered_df = filtered_df.copy()  # 안전하게 복사본 생성한 뒤
    filtered_df['rating'] = np.log1p(filtered_df['play_count'])

    reader = Reader(rating_scale=(filtered_df['rating'].min(), filtered_df['rating'].max()))
    data = Dataset.load_from_df(filtered_df[['user_id', 'title', 'rating']], reader)
    try :
        trainset, testset = train_test_split(data, test_size=0.2, random_state=41)
    except :
        break
    cnt=Eth
    algo = SVD(n_factors=100, n_epochs=20, random_state=42)
    algo.fit(trainset)

    predictions = algo.test(testset)
    
    rating_threshold = filtered_df['rating'].mean()

    precision, recall = precision_recall_at_k_implicit(predictions, k=K, est_threshold=rating_threshold)

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    rows.append(filtered_df['user_id'].count())
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    cnt=Eth
eth_values = list(range(5, cnt+1))

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(10, 6))

color_metrics = 'tab:blue'
color_data = 'tab:green'

# 주 y축: 평가 지표
ax1.set_xlabel('Minimum Listen Count Threshold (Eth)')
ax1.set_ylabel('Evaluation Metrics', color=color_metrics)
ax1.plot(eth_values, precisions, label='Precision@{}'.format(K), marker='o', color='blue')
ax1.plot(eth_values, recalls, label='Recall@{}'.format(K), marker='x', color='orange')
ax1.plot(eth_values, f1_scores, label='F1-Score@{}'.format(K), marker='s', color='red')
ax1.tick_params(axis='y', labelcolor=color_metrics)
ax1.legend(loc='upper left')
ax1.grid(True)

# 보조 y축: 데이터 수 (rows)
ax2 = ax1.twinx()
ax2.set_ylabel('Number of Data', color=color_data)
ax2.plot(eth_values, rows, label='Number of Data', marker='^', color=color_data, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color_data)
ax2.legend(loc='upper right')

plt.title(f'MSD Performance Metrics vs Minimum Listen Count (Top-{K} Recommendation)')
plt.xlim(5, 50)  # x축 범위를 5~50으로 설정
fig.tight_layout()
plt.savefig('MSD_Performance_Metrics_vs_Minimum_Listen_Count.png')
plt.show()