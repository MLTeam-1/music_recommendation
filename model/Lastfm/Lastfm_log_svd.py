# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] SVD 모델 성능 평가 및 '로그 변환(Log Transformation)'의 효과 비교 분석
# [End-to-End 단계] 5. 모델링(Modeling) 및 6. 평가(Evaluation)
#
# [설명]
# 이 스크립트는 추천 시스템의 고전적이면서도 강력한 알고리즘인 SVD(특이값 분해)를 사용하여
# 사용자의 음악 청취 기록(암시적 피드백) 기반의 추천 모델을 구축하고 평가합니다.
#
# 특히, 이 스크립트의 핵심 목표는 'play_count' 데이터를 그대로 사용했을 때와,
# 데이터의 편향을 줄여주는 '로그 변환'을 적용했을 때의 성능 차이를 정량적으로 비교하는 것입니다.
#
# 평가는 두 가지 관점에서 진행됩니다:
#  1. 예측 정확도 (RMSE, MAE): 모델이 사용자의 청취 횟수를 얼마나 정확하게 예측하는가.
#  2. Top-N 추천 성능 (Precision@k, Recall@k): 모델이 추천한 상위 K개의 노래 목록이
#     실제로 사용자가 좋아하는 노래들을 얼마나 잘 포함하고 있는가.
#
#
# K를 10으로 한 이유 : lastfm 데이터는 MSD 데이터와 다르게 한사람당 여러번 10번 이상 듣기 때문에 K가 10개여도 괜찮습니다.
# 따라서 lastfm 데이터에서는 10개를 추천하게 설정했습니다.
# ======================================================================================

import pandas as pd
import numpy as np
from collections import defaultdict

# Surprise 라이브러리: 추천 시스템 알고리즘을 쉽게 테스트할 수 있는 강력한 도구
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# --- 1. 설정 ---
# 이 섹션에서는 모델과 실험에 사용될 주요 파라미터들을 정의합니다.

# 최종 병합된 데이터 파일 경로
FINAL_MERGED_CSV = '../../data/final-lastfm-data.csv'
# Top-N 추천에서 사용할 N값. "상위 몇 개의 아이템을 추천할 것인가?"를 결정합니다.
K = 10  #여러번 해보면서 데이터양을 고려해 선정한 최적의 K
# 최소 청취 횟수 임계값. 이 횟수 미만의 기록을 가진 비활성 사용자는 분석에서 제외합니다.
Eth = 17 # 이거도 K와 같음
# -----------------


# --- 2. 데이터 로딩 및 전처리 ---
# 이 섹션에서는 CSV 파일을 불러와 모델 학습에 적합한 형태로 가공합니다.

try:
    print(f"'{FINAL_MERGED_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(FINAL_MERGED_CSV)
    print("-> 로딩 완료.")
except FileNotFoundError:
    print(f"오류: '{FINAL_MERGED_CSV}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# [핵심 전처리 1] 비활성 사용자 필터링
# 데이터가 너무 적은 사용자는 신뢰할 수 있는 취향 패턴을 학습하기 어렵습니다.
# 따라서, 최소 청취 기록(Eth)을 기준으로 활동적인 사용자만 남겨 모델의 성능과 안정성을 높입니다.
print(f"청취횟수가 {Eth}회 이상인 유저만 포함")
user_counts = df['user_id'].value_counts()
active_users = user_counts[user_counts >= Eth].index
df = df[df['user_id'].isin(active_users)]  
print(f"전체 데이터 수 : {df['user_id'].count()}")  

# [핵심 전처리 2] 로그 변환
# 'play_count'는 소수의 인기곡에 매우 편중된 분포(Long-tail)를 가집니다.
# np.log1p (log(1+x)) 변환을 통해 이 분포를 정규분포에 가깝게 만들어,
# 모델이 극단적인 값에 과도하게 영향을 받는 것을 방지합니다.
# 'rating'이라는 새로운 컬럼에 로그 변환된 값을 저장하여 원본과 비교할 것입니다.
df['rating'] = np.log1p(df['play_count'])
# -----------------


# --- 3. Surprise 데이터셋 생성 및 분리 ---
# 이 섹션에서는 Pandas DataFrame을 Surprise 라이브러리가 이해할 수 있는 데이터 구조로 변환하고,
# 모델 학습용과 평가용으로 데이터를 분리합니다.

# [원본 데이터용] Reader는 평점의 최소-최대 범위를 Surprise에 알려주는 역할을 합니다.
reader = Reader(rating_scale=(df['play_count'].min(), df['play_count'].max()))
data = Dataset.load_from_df(df[['user_id', 'title', 'play_count']], reader)

# [로그 변환 데이터용] 동일하게 로그 변환된 'rating'의 범위를 알려줍니다.
reader_log = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
data_log = Dataset.load_from_df(df[['user_id', 'title', 'rating']], reader_log)

print("\n데이터를 Train Set (80%)과 Test Set (20%)으로 분리합니다...")
# [핵심] random_state를 고정하여, 두 모델(원본, 로그)이 정확히 동일한 데이터로
# 학습하고 평가받도록 함으로써, 비교의 공정성을 확보합니다.
trainset, testset = train_test_split(data, test_size=0.2, random_state=41)
trainset_log, testset_log = train_test_split(data_log, test_size=0.2, random_state=41)

print("-> 데이터 분리 완료.")
# -----------------


# --- 4. 모델 학습 ---
# 이 섹션에서는 분리된 Train Set을 사용하여 두 가지 버전의 SVD 모델을 각각 학습시킵니다.

print("\nTrain Set을 사용하여 SVD 모델을 학습합니다...")
# 모델 인스턴스 생성. random_state를 고정하여 학습 과정의 무작위성을 제어합니다.
algo_for_evaluation = SVD(n_factors=100, n_epochs=20, random_state=42)
algo_for_evaluation_log = SVD(n_factors=100, n_epochs=20, random_state=42)

# fit() 메소드를 통해 실제 학습을 수행합니다. 이 과정에서 모델은 사용자와 아이템의 잠재 요인을 학습합니다.
algo_for_evaluation.fit(trainset)       # 원본 데이터로 학습
algo_for_evaluation_log.fit(trainset_log) # 로그 변환 데이터로 학습

print("-> 모델 학습 완료.")
# -----------------


# --- 5. Top-N 추천 성능 평가 함수 정의 ---
# 이 섹션에서는 추천 시스템의 '랭킹' 성능을 측정하는 핵심 지표인
# Precision@k와 Recall@k를 계산하는 함수를 정의합니다.

def precision_recall_at_k_implicit(predictions, k=10):
    """
    암시적 피드백 데이터에 대한 Precision@k와 Recall@k를 계산합니다.
    '좋아요'의 기준은 Test Set에 해당 아이템의 청취 기록이 존재하는가(true_r > 0) 입니다.
    """
    # 각 사용자별로 (예측 평점, 실제 평점) 리스트를 저장할 딕셔너리
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # 예측 평점(est)이 높은 순으로 정렬하여, 모델의 추천 목록을 시뮬레이션
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Test Set에 있는 이 사용자의 '정답' 아이템의 총 개수
        n_rel = sum((true_r > 0) for (_, true_r) in user_ratings)
        
        # 추천 목록 상위 K개 중에 '정답'이 몇 개나 포함되었는지 카운트
        n_rel_and_rec_k = sum(((true_r > 0)) for (_, true_r) in user_ratings[:k])
        
        # Precision@k: "추천한 K개 중, 몇 개가 진짜 정답이었나?" (정확도)
        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        # Recall@k: "전체 정답 중, K개의 추천 목록이 몇 개나 찾아냈나?" (재현율)
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # 모든 사용자의 평균 Precision과 Recall을 반환
    return sum(p for p in precisions.values()) / len(precisions), \
           sum(r for r in recalls.values()) / len(recalls)
# -----------------


# --- 6. 모델 성능 종합 평가 ---
# 이 섹션에서는 학습된 두 모델을 Test Set에 적용하여 성능을 최종 평가하고 결과를 비교합니다.

# [핵심] model.test()는 학습 과정에서 전혀 사용되지 않은 Test Set을 입력으로 받아,
# 각 (사용자, 아이템) 쌍에 대한 예측 평점을 생성합니다. -> 데이터 유출(Leakage) 없음.
predictions = algo_for_evaluation.test(testset)
predictions_log = algo_for_evaluation_log.test(testset_log)

print("\n" + "="*60)
print("▶ 최종 모델 성능 평가 결과 (SVD: Log Transformation 효과 비교)")
print("="*60)

# 6-a. 예측 정확도 지표 (RMSE, MAE)
# 이 지표들은 모델이 'play_count' 또는 'rating' 값을 얼마나 정확하게 예측하는지를 나타냅니다.
# 참고: 두 모델의 RMSE는 스케일이 달라 직접적인 비교는 무의미할 수 있습니다.
print("\n[1. 예측 정확도 지표 (without log)]\n")
accuracy.rmse(predictions, verbose=True)
accuracy.mae(predictions, verbose=True)

print("\n[1. 예측 정확도 지표 (with log)]\n")
accuracy.rmse(predictions_log, verbose=True)
accuracy.mae(predictions_log, verbose=True)

# 6-b. Top-N 추천 성능 지표 (Precision@k, Recall@k)
# 이 지표들이야말로 '추천 목록의 품질'을 나타내는 핵심적인 성능 척도입니다.
# 두 모델 간의 직접적인 성능 비교는 이 지표들을 통해 이루어져야 합니다.
rating_threshold = df['play_count'].mean()
rating_threshold_log = df['rating'].mean()
print(f"\n[2. Top-{K} 추천 성능 지표 (참고용 임계값: 원본 {rating_threshold:.2f}, 로그 {rating_threshold_log:.2f})]\n")

precision, recall = precision_recall_at_k_implicit(predictions, k=K)
precision_log, recall_log = precision_recall_at_k_implicit(predictions_log, k=K)

print(f"Precision@{K} : Without log : {precision:.4f}, With log :{precision_log:.4f}")
print(f"Recall@{K}    : Without log : {recall:.4f}, With log : {recall_log:.4f}")

# F1-Score는 Precision과 Recall의 조화 평균으로, 두 지표를 종합적으로 보여줍니다.
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

f1_score_log = 2 * (precision_log * recall_log) / (precision_log + recall_log) if (precision_log + recall_log) != 0 else 0

print(f"F1-Score@{K}  : Without log : {f1_score:.4f}, With log : {f1_score_log:.4f}")
# ======================================================================================