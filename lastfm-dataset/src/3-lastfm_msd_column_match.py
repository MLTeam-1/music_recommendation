# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] Last.fm 데이터셋의 컬럼 이름을 MSD 데이터셋 형식과 통일
# [End-to-End 단계] 3. 데이터 정제/전처리(Preprocessing)
#
# [설명]
# 이 스크립트는 정제된 Last.fm 데이터('2.5-lastfm_cleaned.csv')를 입력으로 받습니다.
# 현재 Last.fm 데이터의 컬럼명('userid', 'artname', 'traname')은
# MSD 데이터의 컬럼명('user_id', 'artist', 'title')과 다릅니다.
# 향후 두 데이터셋을 함께 다루거나 코드를 재사용할 때 발생할 수 있는 혼동을 방지하고
# 데이터의 일관성을 유지하기 위해, Last.fm의 컬럼명을 MSD 형식에 맞춰 변경하는 것을
# 목표로 합니다.
# ======================================================================================

import pandas as pd

# --- 1. 데이터 로드 ---
# 이전 단계(2.5단계)에서 생성된, 깨끗하게 정제된 Last.fm 데이터를 로드합니다.
lastfm_df = pd.read_csv('../result/2.5-lastfm_cleaned.csv')

# --- 2. 핵심 로직: 컬럼명 변경 ---
# 이 블록은 Pandas의 rename 기능을 사용하여 지정된 컬럼들의 이름을 한 번에 변경합니다.

# 변경할 컬럼을 매핑하는 딕셔너리를 작성합니다.
# Key: '기존 컬럼명', Value: '새로운 컬럼명'
rename_dict = {
    'userid': 'user_id',  # 'userid'를 'user_id'로 변경
    'artname': 'artist',   # 'artname'을 'artist'로 변경
    'traname': 'title'     # 'traname'을 'title'로 변경
}

# .rename() 함수에 위에서 정의한 딕셔너리를 전달하여 컬럼명을 변경합니다.
# 'columns=' 인자를 사용하여 열(column)의 이름을 변경할 것임을 명시합니다.
lastfm_df = lastfm_df.rename(columns=rename_dict)

# --- 3. 최종 결과 저장 ---
# 컬럼명이 통일된 DataFrame을 새로운 CSV 파일로 저장합니다.
lastfm_df.to_csv('../../data/final-lastfm-data.csv', index=False, encoding='utf-8-sig')

# 작업 완료 메시지를 출력합니다.
print("컬럼명 변경 완료. saved as 'lastfm_matched_msd_columns.csv'")