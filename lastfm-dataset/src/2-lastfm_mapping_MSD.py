# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] Last.fm 청취 기록과 MSD 노래-태그 정보를 '텍스트' 기준으로 병합
# [End-to-End 단계] 3. 데이터 정제/전처리(Preprocessing) 및 데이터 통합(Integration)
#
# [설명]
# 이 스크립트는 ID 체계가 서로 다른 두 데이터셋(Last.fm, MSD)을 연결하는 핵심적인 역할을 합니다.
# 각 데이터셋의 '아티스트명'과 '곡명'을 결합하고, 정규 표현식(regex)을 사용하여
# 괄호 안 내용, 특수 문자 등을 제거한 '정제된 텍스트 키(key)'를 생성합니다.
# 이 공통의 'key'를 기준으로 두 데이터셋을 병합(inner join)하여,
# Last.fm의 청취 기록에 MSD의 풍부한 장르(태그) 정보를 결합하는 것을 목표로 합니다.
# 이 과정을 통해, 최종적으로 모델링에 사용할 통합된 데이터셋을 생성합니다.
# ======================================================================================

import pandas as pd
import re # 정규 표현식(Regular Expression) 처리를 위한 라이브러리

# --- 1. 텍스트 정제 함수 정의 ---
# 이 함수는 문자열을 일관된 형식으로 만들어, 두 데이터셋 간의 매칭 성공률을 높이는 역할을 합니다.
def clean_text(text):
    # 입력된 텍스트를 문자열로 변환하고 모두 소문자로 통일
    text = str(text).lower()
    # 정규 표현식: 괄호 '()'와 그 안의 모든 내용을 삭제 (e.g., "song (remix)" -> "song ")
    text = re.sub(r'\([^)]*\)', '', text)
    # 정규 표현식: 알파벳 소문자(a-z), 숫자(0-9), 공백을 제외한 모든 문자를 제거
    text = re.sub(r'[^a-z0-9 ]', '', text)
    # 정규 표현식: 하나 이상의 연속된 공백을 단 하나의 공백으로 변경
    text = re.sub(r'\s+', ' ', text)
    # 문자열 앞뒤의 불필요한 공백을 제거하고 최종 결과 반환
    return text.strip()

# --- 2. 데이터 로드 ---
# 이전 단계(1단계)에서 생성된, 청취 횟수가 집계된 Last.fm 데이터
lastfm_df = pd.read_csv('../result/1-lastfm_1k_playcount.csv')
# MSD 데이터 처리 과정에서 생성된, 장르가 원-핫 인코딩된 데이터
msd_df = pd.read_csv('../../MSD-dataset/result/2-msd_tags_onehot.csv')


# --- 3. 핵심 로직: 정제된 텍스트 키(Key) 생성 ---
# 이 블록은 두 데이터셋을 연결할 공통의 '다리'를 놓는 과정입니다.
print("두 데이터셋을 병합하기 위한 '정제된 텍스트 키'를 생성합니다...")

# Last.fm 데이터에 'artname'과 'traname'을 합쳐서 'key' 컬럼 생성
# .fillna('')는 비어있는 값(NaN)으로 인한 오류를 방지합니다.
# .map(clean_text)는 'key' 컬럼의 모든 값에 대해 위에서 정의한 clean_text 함수를 적용합니다.
lastfm_df['key'] = (lastfm_df['artname'].fillna('') + ' - ' + lastfm_df['traname'].fillna('')).map(clean_text)

# MSD 데이터에도 'artist'와 'title'을 합쳐서 동일한 방식으로 'key' 컬럼 생성
msd_df['key'] = (msd_df['artist'].fillna('') + ' - ' + msd_df['title'].fillna('')).map(clean_text)
print("-> 'key' 생성 완료.")

# --- 4. 데이터 병합 (Merge) ---
# MSD 데이터에서 song_id가 없는 유효하지 않은 행을 미리 제거합니다.
msd_df = msd_df[msd_df['song_id'].notna() & (msd_df['song_id'] != '')]

print("\n'key'를 기준으로 두 데이터셋을 병합(inner join)합니다...")
# pd.merge를 사용하여 두 DataFrame을 'key' 컬럼을 기준으로 병합합니다.
# how='inner': 'key' 값이 양쪽 DataFrame에 모두 존재하는 행만 남깁니다.
# 즉, MSD에 장르 정보가 있는 노래의 Last.fm 청취 기록만 최종 결과에 포함됩니다.
merged_df = pd.merge(lastfm_df, msd_df, on='key', how='inner')
print("-> 병합 완료.")

# --- 5. 최종 결과 정리 및 저장 ---
# 병합된 결과에서 최종적으로 필요한 컬럼만 선택하여 정리합니다.
# Last.fm의 기본 정보('userid', 'artname', 'traname', 'play_count')와
# MSD에서 가져온 장르 정보(원-핫 인코딩된 컬럼들)를 선택합니다.
# .difference()는 MSD 컬럼 목록에서 불필요한 메타데이터 컬럼들을 제외한 나머지(즉, 장르 컬럼들)를 선택합니다.
result_cols = ['userid', 'artname', 'traname', 'play_count'] + list(msd_df.columns.difference(['song_id','artist', 'title', 'year', 'key']))
result_df = merged_df[result_cols]

# 최종 결과물을 CSV 파일로 저장합니다.
result_df.to_csv('../result/2-lastfm_msd_matched_filtered.csv', index=False, encoding='utf-8-sig')

print(f"\n매칭 및 필터링 완료! 총 {len(result_df)}개 곡이 MSD와 일치합니다.")
print("결과는 'lastfm_msd_matched_filtered.csv'에 저장되었습니다.")