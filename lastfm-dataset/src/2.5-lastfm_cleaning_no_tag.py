# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] Last.fm-MSD 병합 데이터에서 유효하지 않은 기록 제거
# [End-to-End 단계] 3. 데이터 정제/전처리(Preprocessing)
#
# [설명]
# 이 스크립트는 이전 단계에서 생성된 '2-lastfm_msd_matched_filtered.csv' 파일을 입력으로 받습니다.
# 이 파일에는 텍스트 기반 병합 과정에서, 매칭은 되었지만 유효한 장르 정보가 없는
# 노래의 청취 기록이 포함될 수 있습니다. (예: MSD 데이터에 해당 노래의 태그가 없었던 경우)
# 이 스크립트의 목표는 이러한 '장르 정보가 없는' 청취 기록들을 모두 제거하여,
# 최종적으로 모든 행이 유효한 장르 정보를 가지도록 데이터를 정제하는 것입니다.
# ======================================================================================

import pandas as pd

# --- 1. 설정 ---
# 이전 단계(2단계)에서 생성된, Last.fm-MSD 병합 파일을 입력으로 사용합니다.
INPUT_CSV = '../result/2-lastfm_msd_matched_filtered.csv' 

# 이 스크립트의 최종 결과물인, 깨끗하게 정제된 파일의 경로입니다.
OUTPUT_CSV = '../result/2.5-lastfm_cleaned.csv' 
# -----------------

try:
    # --- 2. 데이터 로드 ---
    print(f"'{INPUT_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(INPUT_CSV)
    
    rows_before = len(df)
    print(f"-> 로드 완료. 총 {rows_before}개의 행(청취 기록)이 있습니다.")

    # --- 3. 핵심 로직: 유효 데이터 필터링 ---
    # 이 블록은 각 청취 기록(행)에 대해, 할당된 장르가 하나라도 있는지 검사하고,
    # 장르가 없는 기록은 데이터셋에서 제거합니다.

    # 3-1. 장르 컬럼 목록 자동 식별
    # 메타데이터 컬럼을 명확히 정의하고, 이를 제외한 모든 컬럼을 장르 컬럼으로 간주합니다.
    # 이렇게 하면 나중에 장르 체계가 바뀌어도 코드를 수정할 필요가 없습니다.
    metadata_cols = ['userid','artname','traname','play_count']
    existing_metadata_cols = [col for col in metadata_cols if col in df.columns]
    genre_cols = [col for col in df.columns if col not in existing_metadata_cols]
    print(f"\n총 {len(genre_cols)}개의 장르/태그 컬럼을 기준으로 검사를 시작합니다.")

    # 3-2. 모든 장르 컬럼의 합이 0인 행(기록) 제거
    # 각 행(row)에 대해 장르 컬럼들의 값을 모두 더합니다.
    tag_sum_per_row = df[genre_cols].sum(axis=1)
    
    # 합계가 0보다 큰 행만 남깁니다 (즉, 하나 이상의 장르 태그를 가진 기록만 선택).
    # .copy()를 사용하여 나중에 발생할 수 있는 경고(Warning)를 방지합니다.
    df_cleaned = df[tag_sum_per_row > 0].copy()
    
    rows_after = len(df_cleaned)
    rows_removed = rows_before - rows_after
    
    # [참고] 'year' 컬럼은 이전 병합 단계에서 이미 제외되었으므로, 별도로 제거할 필요가 없습니다.
    # 하지만 코드의 안정성을 위해 확인 후 제거하는 로직은 좋은 습관입니다.
    if 'year' in df_cleaned.columns:
        df_cleaned.drop(columns=['year'], inplace=True)
        print("\n'year' 컬럼을 성공적으로 제거했습니다.")
    else:
        print("\n'year' 컬럼이 존재하지 않아 제거 작업을 건너뜁니다.")

    # --- 4. 결과 요약 및 저장 ---
    print("\n--- 필터링 결과 ---")
    print(f"원본 데이터 행 수: {rows_before}")
    print(f"정리 후 데이터 행 수: {rows_after}")
    print(f"제거된 행 수: {rows_removed} (장르 정보가 없는 기록)")

    # 정제된 데이터를 새로운 CSV 파일로 저장합니다.
    # 제거된 행이 있는 경우에만 파일을 새로 생성하여 불필요한 디스크 공간 낭비를 막습니다.
    if rows_removed > 0:
        print(f"\n'{OUTPUT_CSV}' 파일로 정제된 데이터를 저장하는 중...")
        df_cleaned.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"-> 성공! '{OUTPUT_CSV}' 파일이 생성되었습니다.")
    else:
        # 만약 모든 행이 유효한 장르 정보를 가지고 있었다면, 굳이 새 파일을 만들 필요가 없습니다.
        print("\n제거할 행이 없어 새로운 파일을 생성하지 않았습니다.")

except FileNotFoundError:
    print(f"오류: 입력 파일 '{INPUT_CSV}'을(를) 찾을 수 없습니다.")
    print("이전 단계의 스크립트가 정상적으로 실행되었는지 확인해주세요.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")