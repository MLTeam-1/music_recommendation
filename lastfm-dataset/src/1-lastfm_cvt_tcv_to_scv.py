# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] Last.fm-1k 원본 데이터에서 사용자별/노래별 청취 횟수 계산
# [End-to-End 단계] 3. 데이터 정제/전처리(Preprocessing)
#
# [설명]
# 이 스크립트는 Last.fm-1k의 원본 청취 기록인 TSV(탭으로 구분된) 파일을 입력으로 받습니다.
# 이 파일에는 동일한 사용자가 동일한 노래를 여러 번 들은 기록이 중복으로 포함되어 있습니다.
# 이 스크립트의 목표는 이러한 중복 기록들을 하나로 합치고,
# 'play_count'라는 새로운 컬럼에 총 청취 횟수를 집계하여 저장하는 것입니다.
# 또한, 모델링에 불필요한 'timestamp'와 'traid' 같은 컬럼을 제거하여 데이터를 경량화합니다.
# 이 과정을 통해, 이후 추천 모델에서 사용자의 선호도 강도를 반영할 수 있는
# 핵심적인 데이터를 생성합니다.
# ======================================================================================

import pandas as pd

# --- 1. 설정 ---
# 원본 데이터인 TSV 파일의 경로
input_tsv = '../data/userid-timestamp-artid-artname-traid-traname.tsv'
# 이 스크립트의 최종 결과물인 CSV 파일의 경로
output_csv = '../result/1-lastfm_1k_playcount.csv'

# --- 2. 데이터 로드 및 기본 전처리 ---
# 이 블록은 대용량 TSV 파일을 읽어와 Pandas DataFrame으로 변환하고,
# 불필요한 컬럼을 제거하는 작업을 수행합니다.

# TSV 파일의 각 열(column)에 해당하는 이름을 리스트로 정의합니다.
cols = ['userid', 'timestamp', 'artid', 'artname', 'traid', 'traname']

# Pandas의 read_csv 함수를 사용하여 TSV 파일을 로드합니다.
# sep='\t': 각 데이터가 탭(tab) 문자로 구분되어 있음을 명시합니다.
# names=cols: 파일에 헤더(컬럼명)가 없으므로, 위에서 정의한 'cols' 리스트를 컬럼명으로 사용합니다.
# header=None: 파일의 첫 번째 줄을 데이터로 취급하고, 헤더로 사용하지 않도록 합니다.
# on_bad_lines='skip': 간혹 형식이 잘못된 줄이 있을 경우, 해당 줄을 무시하고 계속 진행합니다.
df = pd.read_csv(input_tsv, sep='\t', names=cols, header=None, on_bad_lines='skip')

# 'timestamp'(청취 시각)와 'traid'(MusicBrainz 트랙 ID) 컬럼은
# 이번 분석에서는 사용하지 않으므로, .drop() 함수를 사용하여 제거합니다.
df = df.drop(columns=['timestamp', 'traid'])

# --- 3. 핵심 로직: 청취 횟수 집계 ---
# 이 블록이 이 스크립트의 가장 중요한 부분입니다.
# 동일한 사용자-노래 조합에 대한 여러 개의 행을 하나로 합치고, 청취 횟수를 계산합니다.

# .groupby() 함수는 지정된 컬럼들의 값이 동일한 행들을 하나의 그룹으로 묶어줍니다.
# 여기서는 'userid', 'artid', 'artname', 'traname'이 모두 동일한 행들이 같은 그룹이 됩니다.
# .size()는 각 그룹의 크기(즉, 행의 개수 = 청취 횟수)를 계산합니다.
# .reset_index(name='play_count')는 그룹화된 결과를 다시 DataFrame으로 변환하고,
# 계산된 횟수 값이 담긴 새로운 컬럼의 이름을 'play_count'로 지정합니다.
df_grouped = df.groupby(['userid', 'artid', 'artname', 'traname']).size().reset_index(name='play_count')

# --- 4. 최종 결과 저장 ---
# 집계가 완료된 DataFrame을 최종 결과물인 CSV 파일로 저장합니다.
# index=False: DataFrame의 인덱스가 파일에 저장되지 않도록 합니다.
# encoding='utf-8-sig': 아티스트나 노래 제목에 포함될 수 있는 한글이나 특수문자가 깨지지 않도록 합니다.
df_grouped.to_csv(output_csv, index=False, encoding='utf-8-sig')

# 작업 완료 메시지와 함께, 최종적으로 생성된 데이터의 행(그룹) 수를 출력합니다.
print(f"변환 완료: {output_csv}, 총 {len(df_grouped)}개의 그룹")```