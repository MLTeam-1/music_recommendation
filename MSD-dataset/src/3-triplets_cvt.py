# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] MSD의 대용량 청취 기록(train_triplets.txt)을 작은 CSV 파일들로 분할
# [End-to-End 단계] 3. 데이터 정제/전처리(Preprocessing)
#
# [설명]
# 이 스크립트는 수천만 라인에 달하는 'train_triplets.txt' 파일을 입력으로 받습니다.
# 이 파일은 크기가 매우 커서 한 번에 메모리에 로드하면 시스템에 큰 부담을 줄 수 있습니다.
# 따라서, Pandas의 'chunksize' 옵션을 사용하여 파일을 지정된 크기의 여러 조각(chunk)으로 나누어
# 각각 별도의 CSV 파일로 저장하는 것을 목표로 합니다.
# 이 과정을 통해, 이후 단계에서 데이터를 메모리 문제 없이 효율적으로 처리할 수 있게 됩니다.
# ======================================================================================

import pandas as pd
import os

# --- 1. 설정 ---
# 원본 Taste Profile Subset 파일 경로
INPUT_TXT_FILE = '../data/train_triplets.txt' 

# 분할하여 저장할 파일들의 경로와 이름 접두사
OUTPUT_PREFIX = '../result/3-triplits/taste_profile_part_' 

# 각 CSV 파일에 저장할 최대 줄(row)의 수를 지정합니다.
# 이 값을 조절하여 분할될 파일의 크기와 개수를 결정할 수 있습니다.
ROWS_PER_CHUNK = 1000000 # 백만 줄 단위로 분할
# -----------------

# --- 2. 핵심 로직: 파일 분할 ---
# 이 블록은 대용량 파일을 메모리에 모두 올리지 않고, 조금씩 읽어서 처리하는
# 메모리 효율적인(memory-efficient) 방식으로 동작합니다.

# 원본 파일이 존재하는지 먼저 확인하여 오류를 방지합니다.
if not os.path.exists(INPUT_TXT_FILE):
    print(f"오류: 원본 파일 '{INPUT_TXT_FILE}'을(를) 찾을 수 없습니다.")
else:
    try:
        print(f"'{INPUT_TXT_FILE}' 파일을 {ROWS_PER_CHUNK:,}줄 단위로 분할하여 CSV로 저장합니다.")
        
        # 생성될 파일의 번호를 매기기 위한 카운터 변수
        chunk_number = 1
        
        # [핵심] pd.read_csv에 'chunksize' 옵션을 사용하면, 전체 파일을 한 번에 읽지 않습니다.
        # 대신, 'ROWS_PER_CHUNK' 만큼의 줄을 가진 작은 DataFrame 조각(chunk)을
        # 순서대로 반환하는 '반복자(iterator)'를 생성합니다.
        for chunk in pd.read_csv(
            INPUT_TXT_FILE, 
            sep='\t',          # 파일의 각 컬럼은 탭(\t)으로 구분되어 있음
            header=None,       # 파일의 첫 줄에 헤더(컬럼 이름)가 없음
            names=['user_id', 'song_id', 'play_count'], # 헤더가 없으므로 직접 컬럼 이름 지정
            chunksize=ROWS_PER_CHUNK
        ):
            # 저장할 파일 이름 생성 (예: taste_profile_part_1.csv)
            output_filename = f"{OUTPUT_PREFIX}{chunk_number}.csv"
            
            print(f"  - 처리 중: {chunk_number}번째 조각 -> '{output_filename}' 파일로 저장...")
            
            # 현재 읽어들인 'chunk' (작은 DataFrame)를 CSV 파일로 저장
            chunk.to_csv(output_filename, index=False, encoding='utf-8-sig')
            
            # 다음 파일 번호를 위해 카운터를 1 증가시킵니다.
            chunk_number += 1

        print(f"\n-> 성공! 총 {chunk_number - 1}개의 파일로 분할하여 저장을 완료했습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")