# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] MSD 데이터셋에서 '순수' 장르 매니아 사용자 분포 분석
# [End-to-End 단계] 4. 데이터 탐색(Exploratory Data Analysis, EDA)
#
# [설명]
# 이 스크립트는 최종 병합된 MSD 데이터셋을 입력으로 받습니다.
# 각 사용자의 전체 청취 기록을 바탕으로 개별 '취향 프로필'(장르 선호도 비율)을 계산합니다.
# 그 후, 프로필을 분석하여 특정 장르의 선호도가 거의 100%에 달하는,
# 즉 한 가지 장르만 집중적으로 소비하는 '순수 매니아' 사용자가 각 장르별로 몇 명이나 되는지
# 집계하는 것을 목표로 합니다.
# 이를 통해 사용자 취향의 편중 현상을 파악하고, 데이터의 특성을 더 깊이 이해할 수 있습니다.
# ======================================================================================

import pandas as pd
from tqdm import tqdm # 오래 걸리는 작업의 진행 상황을 보여주는 progress bar

# --- 1. 설정 ---
# 최종 병합된 데이터 파일 경로 (MSD 데이터 중 첫 번째 파트)
FINAL_MERGED_CSV = '../result/4-merged_csv/final_data_sorted_part_1.csv' 

# --- 2. 데이터 로딩 ---
try:
    print(f"'{FINAL_MERGED_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(FINAL_MERGED_CSV)
    print("-> 로딩 완료.")
except FileNotFoundError:
    print(f"오류: '{FINAL_MERGED_CSV}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# --- 3. 장르 컬럼 식별 ---
# 분석에 사용할, 이전에 정의했던 대분류 장르 목록
main_genres = [
    'Rock', 'Classic Rock', 'Hard Rock', 'Alternative & Indie Rock', 'Pop & Folk Rock', 
    'Pop', 'Jazz & Blues', 'R&B & Funk', 'Hip Hop', 'Electronic & Dance', 
    'Folk & Country', 'Reggae', 'Other'
]
# 실제 데이터프레임에 존재하는 장르 컬럼만 필터링하여 사용
existing_genre_cols = [col for col in main_genres if col in df.columns]

if not existing_genre_cols:
    print("오류: 데이터에서 장르 컬럼을 찾을 수 없습니다. 컬럼명을 확인해주세요.")
    exit()

# --- 4. 핵심 로직: '순수' 장르 매니아 사용자 분석 ---
# 이 블록은 먼저 모든 사용자의 취향 프로필을 계산한 뒤,
# 각 프로필을 검사하여 한 장르의 선호도가 100%인 사용자를 카운트합니다.

print("\n각 사용자의 장르 프로필을 분석하여 '순수' 장르 매니아를 찾는 중입니다...")

# 각 장르별 '순수 매니아' 수를 저장할 딕셔너리를 0으로 초기화
pure_genre_counts = {genre: 0 for genre in existing_genre_cols}
user_profiles = {} # 계산된 사용자 프로필을 재사용하기 위해 저장할 딕셔너리

# [핵심 계산 1] 사용자별 프로필 계산
# 이 과정은 이전 EDA 코드와 동일합니다. 'user_id'로 그룹화하여 가중 평균을 계산합니다.
for user_id, user_data in tqdm(df.groupby('user_id'), desc="Calculating Profiles"):
    user_genre_data = user_data[existing_genre_cols]
    play_counts = user_data['play_count']
    
    weighted_genre_sum = user_genre_data.mul(play_counts, axis=0).sum()
    total_play_count = play_counts.sum()
    
    if total_play_count > 0:
        user_profile_vector = weighted_genre_sum / total_play_count
        user_profiles[user_id] = user_profile_vector

# [핵심 계산 2] 계산된 프로필을 바탕으로 '순수' 매니아 카운트
for user_id, profile in user_profiles.items():
    # 프로필의 합이 0인 경우 (예: 태그 없는 노래만 들은 경우)는 분석에서 제외
    if profile.sum() == 0:
        continue
        
    # [판단 로직] 어떤 장르의 선호도가 100%에 가까운지 확인
    # (profile >= 0.999)는 각 장르 선호도가 0.999 이상이면 True, 아니면 False를 반환합니다.
    # .sum() == 1은 True가 정확히 한 개만 있는지를 확인하는 것입니다.
    is_pure_fan = (profile >= 0.999).sum() == 1
    
    if is_pure_fan:
        # 만약 '순수 매니아'가 맞다면,
        # .idxmax()는 그 프로필에서 가장 높은 값을 가진 장르의 이름(인덱스)을 찾아냅니다.
        pure_genre = profile.idxmax()
        # 해당 장르의 카운트를 1 증가시킵니다.
        if pure_genre in pure_genre_counts:
            pure_genre_counts[pure_genre] += 1

# --- 5. 결과 출력 ---
total_users = len(user_profiles)
print("\n" + "="*60)
print(f"▶ 분석 결과 (총 {total_users}명의 사용자 대상)")
print("="*60)

# 결과를 Pandas Series로 변환하여 값(count)이 높은 순으로 정렬
pure_fan_series = pd.Series(pure_genre_counts).sort_values(ascending=False)

print("[오직 한 가지 장르만 100% 선호하는 사용자 수]\n")

# 정렬된 결과를 순회하며, 카운트가 0보다 큰 장르만 출력
for genre, count in pure_fan_series.items():
    if count > 0:
        percentage = (count / total_users) * 100
        print(f"- {genre}: {count} 명 ({percentage:.2f}%)")

# 이 분석을 통해 얻은 인사이트나 가설을 검증하는 부분
print("\n" + "-"*60)
print("[가설 검증]")
rock_pure_fans = pure_fan_series.get('Rock', 0) 
if rock_pure_fans > 0:
    rock_percentage = (rock_pure_fans / total_users) * 100
    print(f"-> 'Rock' 장르에만 100% 편향된 사용자는 총 {rock_pure_fans}명으로,")
    print(f"   전체 사용자의 약 {rock_percentage:.2f}%를 차지합니다.")
else:
    print("-> 'Rock' 계열에만 100% 편향된 사용자는 거의 없는 것으로 보입니다.")