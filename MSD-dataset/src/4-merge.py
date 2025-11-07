# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] MSD의 분할된 청취 기록과 노래-태그 정보를 최종 병합
# [End-to-End 단계] 3. 데이터 정제/전처리(Preprocessing) 및 5. 특성 공학(Feature Engineering)
#
# [설명]
# 이 스크립트는 '양(Quantity)' 중심 데이터셋을 완성하는 최종 단계입니다.
# 1. 이전 단계에서 생성된 분할된 청취 기록 파일들(taste_profile_part_*.csv)을 순차적으로 읽습니다.
# 2. 각 청취 기록을 노래-태그 정보 파일(2.5-msd_cleaned.csv)과 'song_id'를 기준으로 병합합니다.
# 3. 병합된 데이터들을 'user_id'를 기준으로 다시 그룹화하고 정렬하여, 사용자별 데이터가 흩어지지 않도록 합니다.
# 4. 최종적으로, 정렬된 데이터를 다시 100만 행 단위의 여러 파일로 나누어 저장합니다.
# 이 과정을 통해 모델링에 바로 사용할 수 있는, 깨끗하고 정렬된 최종 데이터셋을 생성합니다.
# ======================================================================================

import pandas as pd
import os
import glob
from collections import defaultdict

# --- 1. 설정 ---
# [입력 1] 이전 단계(2.5단계)에서 생성된, 노래-태그 정보 파일
TAGS_FILE = '../result/2.5-msd_cleaned.csv'

# [입력 2] 이전 단계(3단계)에서 생성된, 분할된 청취 기록 파일들을 찾기 위한 패턴
CHUNK_PATTERN = '../result/3-triplits/taste_profile_part_*.csv'

# [출력] 최종 결과물을 저장할 경로와 이름 접두사
OUTPUT_PREFIX = '../../data/final-msd-data.csv'


# 각 최종 파일에 저장할 최대 행(row)의 수
ROWS_PER_OUTPUT_FILE = 1000000

# -----------------

if not os.path.exists(TAGS_FILE):
    print(f"오류: 메인 태그 파일 '{TAGS_FILE}'을(를) 찾을 수 없습니다.")
else:
    # glob.glob()은 와일드카드(*)를 사용하여 패턴에 맞는 모든 파일 경로를 리스트로 가져옵니다.
    chunk_files = sorted(glob.glob(CHUNK_PATTERN))
    if not chunk_files:
        print(f"오류: 패턴 '{CHUNK_PATTERN}'에 맞는 분할된 파일을 찾을 수 없습니다.")
    else:
        try:
            # --- 2. 핵심 로직 1: 데이터 로드 및 병합 후 그룹화 ---
            # 이 블록은 메모리 사용량을 관리하기 위해, 병합된 데이터를 사용자별로 묶어 딕셔너리에 저장합니다.
            
            # 2-1. 노래-태그 정보 로드
            # 이 정보는 모든 청취 기록 조각(chunk)과 병합되어야 하므로, 메모리에 한 번만 불러옵니다.
            print(f"메인 태그 파일 '{TAGS_FILE}'을(를) 로드합니다...")
            tags_df = pd.read_csv(TAGS_FILE)
            print("-> 태그 파일 로드 완료.")

            # 2-2. 사용자별 데이터를 임시 저장할 딕셔너리
            # defaultdict(list)는 키가 없을 때 자동으로 빈 리스트를 값으로 생성해줍니다.
            user_data_map = defaultdict(list)
            print(f"\n--- 총 {len(chunk_files)}개의 파일에서 데이터를 읽고 병합하며 user_id로 그룹화합니다 ---")
            
            for i, chunk_file in enumerate(chunk_files):
                print(f"  ({i+1}/{len(chunk_files)}) 처리 중: '{os.path.basename(chunk_file)}'")
                # 각 청취 기록 조각과 태그 정보를 'song_id'를 기준으로 병합(merge)합니다.
                # how='inner'는 양쪽에 모두 'song_id'가 존재하는 기록만 남깁니다 (태그 정보가 없는 노래는 제외).
                merged_chunk = pd.merge(pd.read_csv(chunk_file), tags_df, on='song_id', how='inner')
                
                # [핵심] 병합된 결과를 다시 'user_id'로 그룹화하여 딕셔너리에 추가합니다.
                # 이 과정을 통해, 여러 파일에 흩어져 있던 한 사용자의 기록이 같은 키 아래로 모이게 됩니다.
                for user_id, user_group_df in merged_chunk.groupby('user_id'):
                    user_data_map[user_id].append(user_group_df)
            
            print("-> 모든 데이터 그룹화 완료.")

            # --- 3. 핵심 로직 2: 사용자 ID 정렬 및 순차적 파일 쓰기 ---
            # 이 블록은 그룹화된 데이터를 사용자 ID 순서로 정렬한 뒤,
            # 다시 큰 파일들로 나누어 저장하는 역할을 합니다.

            # 3-1. 사용자 ID 정렬
            print("\n--- 모든 사용자 ID를 정렬합니다 ---")
            sorted_user_ids = sorted(user_data_map.keys())
            print(f"-> 총 {len(sorted_user_ids):,}명의 고유한 사용자를 찾았습니다.")

            print(f"\n--- 정렬된 사용자 순서대로 새 파일을 작성합니다 (파일당 {ROWS_PER_OUTPUT_FILE:,} 행) ---")
            output_buffer = []      # 파일로 쓰기 전 데이터를 임시로 모아두는 버퍼
            current_buffer_rows = 0 # 현재 버퍼에 쌓인 행의 수
            output_file_counter = 1 # 출력 파일 번호

            # 3-2. 정렬된 사용자 순서대로 데이터 처리
            for i, user_id in enumerate(sorted_user_ids):
                # 딕셔너리에 저장된, 한 사용자의 모든 데이터 조각(DataFrame)들을 하나로 합칩니다.
                user_complete_df = pd.concat(user_data_map[user_id], ignore_index=True)
                
                # 합쳐진 사용자 데이터를 버퍼에 추가하고, 행 수를 누적합니다.
                output_buffer.append(user_complete_df)
                current_buffer_rows += len(user_complete_df)

                # [핵심] 버퍼의 행 수가 설정한 임계값(ROWS_PER_OUTPUT_FILE)을 넘거나,
                # 마지막 사용자인 경우, 버퍼의 내용을 파일로 저장합니다.
                if current_buffer_rows >= ROWS_PER_OUTPUT_FILE or (i + 1) == len(sorted_user_ids):
                    output_filename = f"{OUTPUT_PREFIX}"
                    print(f"  -> 버퍼가 임계값에 도달하여 '{output_filename}' 파일로 저장합니다... (현재 행: {current_buffer_rows:,})")
                    
                    # 버퍼에 있는 모든 DataFrame들을 최종적으로 하나로 합친 뒤,
                    final_output_df = pd.concat(output_buffer, ignore_index=True)

                    # 병합의 키로 사용된 'song_id'는 더 이상 필요 없으므로 제거합니다.
                    if 'song_id' in final_output_df.columns:
                        final_output_df.drop(columns=['song_id'], inplace=True)
                        print("'song_id' 컬럼을 성공적으로 제거했습니다.")    
                    
                    # 최종 결과물을 CSV 파일로 저장합니다.
                    final_output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
                    
                    # 다음 파일을 위해 버퍼와 카운터를 초기화합니다.
                    output_buffer = []
                    current_buffer_rows = 0
                    output_file_counter += 1
                
            print("\n==============================================")
            print("모든 데이터의 user_id 기준 정렬 및 재분할 작업이 완료되었습니다!")
            print(f"총 {output_file_counter - 1}개의 정렬된 파일이 생성되었습니다.")
            print("==============================================")

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")