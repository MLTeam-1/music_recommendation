# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] MSD 요약 데이터의 세부 태그를 대분류 마스터 장르로 통합
# [End-to-End 단계] 5. 특성 공학(Feature Engineering)
#
# [설명]
# 이 스크립트는 이전 단계에서 생성된 '1-msd_summary.csv' 파일을 입력으로 받습니다.
# 파일 내의 쉼표로 구분된 'tags' 문자열을 분석하여, 사전에 정의된
# 'genre_mapping' 규칙에 따라 여러 세부 태그들을 소수의 대분류 장르로 통합합니다.
# 최종적으로, 각 노래가 어떤 대분류 장르에 속하는지를 0 또는 1로 표시하는
# 원-핫 인코딩된 형태의 CSV 파일을 생성하는 것을 목표로 합니다.
# ======================================================================================

import pandas as pd

# --- 1. 설정 ---
# 이전 단계(1단계)에서 생성된, 상위 태그만 포함된 요약 파일을 입력으로 사용합니다.
INPUT_CSV = '../result/1-msd_summary.csv'
# 이 스크립트의 최종 결과물인, 대분류 장르로 통합된 파일의 경로입니다.
OUTPUT_CSV = '../result/2-msd_tags_onehot.csv' 

# 너무 드물게 나타나는 태그는 노이즈일 가능성이 높으므로,
# 최소 이 횟수 이상 등장한 태그만 분석 대상으로 삼습니다.
MIN_TAG_FREQUENCY = 200
# -----------------

# --- 2. 대분류 장르 매핑 규칙 ---
# 이 딕셔너리는 '세부 태그'를 '대분류 장르'로 어떻게 매핑할지 정의하는 핵심 규칙입니다.
# Key: 세부 태그 (소문자), Value: 할당될 대분류 장르
genre_mapping = {# 5-check_data_bias로 분포를 계속 확인하면서 그나마 최적의 분포를 위해 이렇게 나눴습니다.
    # Rock 계열을 4개로 세분화하여 표현력을 높임
    'rock': 'Rock',
    'classic rock': 'Classic Rock',
    'hard rock': 'Hard Rock',
    'heavy metal': 'Hard Rock',
    'alternative rock': 'Alternative & Indie Rock',
    'indie rock': 'Alternative & Indie Rock',
    'punk': 'Alternative & Indie Rock',
    'pop rock': 'Pop & Folk Rock',
    'blues-rock': 'Pop & Folk Rock',
    'folk rock': 'Pop & Folk Rock',
    'country rock': 'Pop & Folk Rock',
    'soft rock': 'Pop & Folk Rock',

    # 나머지 장르들도 각각의 대분류로 매핑
    'pop': 'Pop',
    'ballad': 'Pop',
    'easy listening': 'Pop',
    'chanson': 'Pop',
    'blues': 'Jazz & Blues',
    'jazz': 'Jazz & Blues',
    'latin jazz': 'Jazz & Blues',
    'smooth jazz': 'Jazz & Blues',
    'jazz funk': 'Jazz & Blues',
    'r&b': 'R&B & Funk',
    'funk': 'R&B & Funk',
    'hip hop': 'Hip Hop',
    'rap': 'Hip Hop',
    'electronic': 'Electronic & Dance',
    'chill-out': 'Electronic & Dance',
    'disco': 'Electronic & Dance',
    'downtempo': 'Electronic & Dance',
    'trip hop': 'Electronic & Dance',
    'progressive house': 'Electronic & Dance',
    'folk': 'Folk & Country',
    'country': 'Folk & Country',
    'singer-songwriter': 'Folk & Country',
    'reggae': 'Reggae',
    'dub': 'Reggae',
    'dancehall': 'Reggae',
    'roots reggae': 'Reggae',
    'soundtrack': 'Other',
    'ccm': 'Other',
    'los angeles': 'Other'
}
# -----------------------------------------

try:
    # --- 3. 데이터 로드 및 전처리 ---
    print(f"'{INPUT_CSV}' 파일을 읽는 중...")
    df = pd.read_csv(INPUT_CSV)
    # 'tags' 컬럼에 비어있는 값(NaN)이 있을 경우, 오류 방지를 위해 빈 문자열로 채웁니다.
    df['tags'] = df['tags'].fillna('')

    # --- 4. 세부 태그 원-핫 인코딩 (대분류 작업을 위한 임시 단계) ---
    # 이 블록은 먼저 데이터에 자주 등장하는 '고빈도 태그' 목록을 만든 뒤,
    # 각 노래가 이 고빈도 태그를 포함하는지 여부를 0 또는 1로 표시하는 임시 컬럼들을 생성합니다.
    print("고빈도 태그를 수집하고 임시 원-핫 인코딩을 수행하는 중...")
    
    # 전체 데이터에서 각 태그가 몇 번 등장했는지 계산
    tag_counts = pd.Series([tag for tags_list in df['tags'] for tag in tags_list.split(',') if tag]).value_counts()
    # 설정한 최소 등장 횟수(MIN_TAG_FREQUENCY)를 만족하는 태그만 필터링
    frequent_tags = tag_counts[tag_counts >= MIN_TAG_FREQUENCY].index.tolist()
    print(f"{len(frequent_tags)}개의 고빈도 태그를 컬럼으로 사용합니다.")

    # 각 노래의 태그 문자열을 set으로 변환하여 검색 속도를 높임
    song_tags_set = df['tags'].str.split(',').apply(set)
    # 고빈도 태그 목록을 순회하며, 각 태그에 대한 임시 컬럼 생성
    for tag in frequent_tags:
        df[tag] = song_tags_set.apply(lambda x: 1 if tag in x else 0)

    # --- 5. 대분류 장르 컬럼 생성 ---
    # 이 블록이 이 스크립트의 핵심입니다. 앞서 만든 임시 세부 태그 컬럼들을
    # 'genre_mapping' 규칙에 따라 합산하여 최종 대분류 장르 컬럼을 생성합니다.
    print("\n대분류 장르 컬럼을 생성하는 중...")
    
    # 'genre_mapping'에 있는 모든 대분류 장르의 고유 목록을 만듭니다.
    main_genres = sorted(list(set(genre_mapping.values())))
    
    for main_genre in main_genres:
        # 현재 대분류 장르에 속하는 세부 태그들을 매핑 규칙에서 찾아옵니다.
        sub_tags = [tag for tag, m_genre in genre_mapping.items() 
                    if m_genre == main_genre and tag in frequent_tags]
        
        # 만약 해당하는 세부 태그가 있다면,
        if sub_tags:
            # 해당 세부 태그 컬럼들의 값을 행(row) 단위로 모두 더한 뒤,
            # 그 합이 0보다 크면(즉, 세부 태그 중 하나라도 1이면) 1로, 아니면 0으로 설정합니다.
            df[main_genre] = df[sub_tags].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
            print(f" - '{main_genre}' 컬럼 생성 완료.")

    # --- 6. 불필요한 임시 컬럼 삭제 ---
    # 대분류 장르 컬럼 생성이 완료되었으므로, 중간 과정에서 사용된
    # 세부 태그 컬럼들과 원본 'tags' 문자열 컬럼을 모두 삭제하여 최종 결과물을 정리합니다.
    print("\n대분류 작업에 사용된 임시 컬럼들을 삭제하는 중...")
    columns_to_drop = frequent_tags + ['tags']
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"{len(columns_to_drop)}개의 컬럼을 삭제했습니다.")
    
    # --- 7. 최종 결과 저장 ---
    print(f"\n최종 결과를 '{OUTPUT_CSV}' 파일로 저장하는 중...")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n-> 성공! '{OUTPUT_CSV}' 파일이 생성되었습니다.")
    print("\n--- 최종 데이터 샘플 (상위 5개) ---")
    print(df.head())

except FileNotFoundError:
    print(f"오류: 입력 파일 '{INPUT_CSV}'을(를) 찾을 수 없습니다.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")