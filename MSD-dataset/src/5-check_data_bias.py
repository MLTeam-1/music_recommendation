# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] MSD 데이터셋의 전체 사용자 장르 선호도 분포 분석 및 시각화
# [End-to-End 단계] 4. 데이터 탐색(Exploratory Data Analysis, EDA)
#
# [설명]
# 이 스크립트는 최종적으로 정제되고 병합된 MSD 데이터셋 중 첫 번째 파일
# ('final_data_sorted_part_1.csv')을 입력으로 받습니다.
# 각 사용자의 청취 기록과 청취 횟수(play_count)를 가중치로 사용하여,
# 모든 사용자의 '평균적인 취향 프로필'을 계산합니다.
# 이를 통해 "이 데이터셋의 사용자들은 전반적으로 어떤 장르를 가장 선호하는가?"라는
# 질문에 대한 답을 구하고, 그 결과를 막대그래프로 시각화하여 데이터의 특성을
# 한눈에 파악하는 것을 목표로 합니다.
# ======================================================================================

import pandas as pd
import matplotlib.pyplot as plt # 데이터 시각화를 위한 라이브러리
import seaborn as sns          # Matplotlib을 더 예쁘고 쉽게 사용하기 위한 라이브러리
from tqdm import tqdm          # 오래 걸리는 작업의 진행 상황을 보여주는 progress bar

# --- 1. 설정 ---
# 이전 단계(4단계)에서 생성된, 최종 병합된 데이터 파일 경로
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
# 분석에 사용할 마스터 장르 컬럼 목록을 정의하고,
# 실제 데이터프레임에 존재하는 컬럼만 필터링합니다.
main_genres = ['Alternative & Indie Rock','Rock', 'Classic Rock','Hard Rock','Electronic & Dance','Folk & Country','Hip Hop','Jazz & Blues','Other','Pop','Pop & Folk Rock','R&B & Funk','Reggae']
existing_genre_cols = [col for col in main_genres if col in df.columns]

# --- 4. 핵심 로직: 전체 사용자 장르 선호도 분포 계산 ---
# 이 블록은 각 사용자의 개별 취향 프로필을 모두 계산한 뒤,
# 이 프로필들의 평균을 내어 데이터셋 전체의 경향성을 파악합니다.

print("\n전체 사용자의 장르 선호도 분포를 계산하는 중입니다...")

# 4-a. 각 사용자별 '종합 취향 프로필' 계산
# groupby('user_id')를 사용하여 사용자별로 데이터를 묶습니다.
all_user_profiles = [] # 각 사용자의 프로필 벡터를 저장할 리스트

# tqdm을 사용하여 많은 사용자를 처리하는 동안 진행 상황을 시각적으로 보여줍니다.
for user_id, user_data in tqdm(df.groupby('user_id'), desc="Calculating User Profiles"):
    user_genre_data = user_data[existing_genre_cols]
    play_counts = user_data['play_count']
    
    # [핵심 계산] 청취 횟수(play_counts)를 가중치로 사용하여 각 장르의 총점을 계산합니다.
    # .mul()은 각 행의 장르(0 또는 1)에 해당 행의 play_count를 곱합니다.
    # .sum()은 이렇게 계산된 값들을 모두 더하여, 사용자의 장르별 총 청취 횟수를 구합니다.
    weighted_genre_sum = user_genre_data.mul(play_counts, axis=0).sum()
    total_play_count = play_counts.sum()
    
    # 총 청취 횟수가 0보다 큰 경우에만 계산을 수행합니다 (0으로 나누기 방지).
    if total_play_count > 0:
        # 사용자의 '취향 프로필 벡터'를 계산합니다. (각 장르별 청취 비율)
        user_profile_vector = weighted_genre_sum / total_play_count
        all_user_profiles.append(user_profile_vector)

# 4-b. 모든 사용자 프로필을 하나의 DataFrame으로 합칩니다.
all_profiles_df = pd.DataFrame(all_user_profiles)

# 4-c. 모든 프로필의 평균을 계산하여 데이터셋 전체의 평균 선호도 분포를 구합니다.
# .mean()은 각 장르(컬럼)별로 모든 사용자의 선호도 점수 평균을 계산합니다.
# .sort_values()를 통해 가장 선호도가 높은 장르부터 순서대로 정렬합니다.
dataset_genre_distribution = all_profiles_df.mean().sort_values(ascending=False)

print("\n" + "="*50)
print("▶ 데이터셋 전체의 장르 선호도 분포")
print("="*50)
print(dataset_genre_distribution)


# --- 5. 시각화 ---
# 이 블록은 위에서 계산된 분포를 사용자가 한눈에 파악할 수 있도록 막대그래프로 그립니다.
print("\n분포를 막대그래프로 시각화하는 중...")

# 그래프의 크기를 설정합니다 (가로 12인치, 세로 6인치).
plt.figure(figsize=(12, 6))

# seaborn의 barplot을 사용하여 막대그래프를 그립니다.
# x축은 장르 이름, y축은 평균 선호도 점수입니다.
# palette='viridis'는 보기 좋은 색상 조합을 자동으로 적용해줍니다.
sns.barplot(x=dataset_genre_distribution.index, y=dataset_genre_distribution.values, palette='viridis')

# 그래프의 제목과 축 레이블을 설정합니다.
plt.title('Overall Genre Preference Distribution in the Dataset')
plt.ylabel('Average Preference Score (Ratio)')
plt.xlabel('Main Genres')

# x축의 장르 이름들이 길어서 겹칠 수 있으므로, 45도 회전시켜 가독성을 높입니다.
plt.xticks(rotation=45, ha='right')
# 그래프 요소들이 잘리지 않고 잘 보이도록 레이아웃을 조정합니다.
plt.tight_layout()

# 분석 결과를 이미지 파일로 저장합니다.
file_name = '5-dataset_genre_distribution.png'
plt.savefig(file_name)
print(f"-> 분석 그래프가 '{file_name}' 으로 저장되었습니다.")

# Jupyter Notebook 등에서 그래프를 바로 확인하기 위해 화면에 출력합니다.
plt.show()