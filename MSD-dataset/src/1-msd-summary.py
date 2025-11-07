# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] Million Song Dataset(MSD) 원본 데이터 처리 (헬퍼 스크립트 활용)
# [End-to-End 단계] 2. 데이터 수집(Sourcing) 및 3. 데이터 정제/전처리(Preprocessing)
#
# [설명]
# 이 스크립트는 공식 헬퍼 스크립트(hdf5_getters.py)를 사용하여 MSD의 .h5 파일들을 처리합니다.
# 각 파일에서 노래의 핵심 메타데이터(ID, 제목, 아티스트 등)와 함께,
# 아티스트에게 붙은 태그 중 상위 N개, 그리고 MusicBrainz ID(MBID)를 추출하여
# 하나의 요약된 CSV 파일로 저장하는 것을 목표로 합니다.
# ======================================================================================

import os
import pandas as pd
import hdf5_getters as GETTERS # 공식 헬퍼 스크립트를 GETTERS라는 별명으로 불러옴

# --- 1. 설정 ---
# [단계 2: 데이터 수집] 원본 데이터 경로
msd_subset_path = '../data/MillionSongSubset'
# [단계 3: 데이터 정제/전처리] 결과물 저장 경로
output_csv_file = '../result/1-msd_summary_with_mbid.csv'
# 각 노래에서 추출할 상위 태그의 개수를 지정
NUM_TOP_TAGS = 5 
# ------------

# 추출된 모든 노래 정보를 저장할 리스트를 초기화합니다.
all_songs_data = []
print(f"MSD 요약 파일 생성을 시작합니다. (상위 {NUM_TOP_TAGS}개 태그 및 MBID 포함)")
print("시간이 다소 걸릴 수 있습니다...")

# --- 2. 핵심 로직: 모든 .h5 파일 순회 및 데이터 추출 ---
# 이 블록은 'os.walk'를 사용하여 모든 하위 폴더를 탐색하며 .h5 파일을 찾아 처리합니다.
i = 0
for root, dirs, files in os.walk(msd_subset_path):
    for file in files:
        if file.endswith('.h5'):
            file_path = os.path.join(root, file)
            try:
                # 헬퍼 스크립트의 함수를 사용하여 h5 파일을 안전하게 엽니다.
                h5 = GETTERS.open_h5_file_read(file_path)

                # --- 2-1. 메타데이터 추출 ---
                # 헬퍼 함수들을 사용하여 각 필드의 데이터를 추출하고, 바이트 문자열을 일반 문자열로 변환(decode)합니다.
                song_id = GETTERS.get_song_id(h5).decode('utf-8')
                artist = GETTERS.get_artist_name(h5).decode('utf-8')
                title = GETTERS.get_title(h5).decode('utf-8')
                year = GETTERS.get_year(h5)

                # --- 2-2. MusicBrainz ID (MBID) 추출 ---
                # MBID는 일부 파일에만 존재할 수 있으므로, 예외 처리를 통해 안전하게 추출합니다.
                try:
                    mbid_bytes = GETTERS.get_track_mbid(h5)
                    mbid = mbid_bytes.decode('utf-8') if mbid_bytes else ''
                except AttributeError:
                    mbid = '' # get_track_mbid 함수가 없는 구버전 헬퍼 스크립트를 위한 호환성 처리

                # --- 2-3. 상위 N개 태그 추출 ---
                # 아티스트에게 붙은 모든 태그를 가져온 뒤, 설정한 개수(NUM_TOP_TAGS)만큼만 잘라냅니다.
                all_terms = GETTERS.get_artist_terms(h5)
                top_terms = all_terms[:NUM_TOP_TAGS]
                # 잘라낸 태그 리스트를 쉼표(,)로 구분된 하나의 문자열로 만듭니다.
                terms_str = ','.join([term.decode('utf-8') for term in top_terms])

                # --- 2-4. 데이터 취합 ---
                # 추출된 모든 정보를 딕셔너리 형태로 정리합니다.
                song_data = {
                    'song_id': song_id,
                    'mbid': mbid,
                    'artist': artist,
                    'title': title,
                    'year': year,
                    'tags': terms_str
                }
                all_songs_data.append(song_data)

                # 진행 상황을 1000개 파일마다 출력합니다.
                i += 1
                if i % 1000 == 0:
                    print(f"{i}개의 파일을 처리했습니다...")

                h5.close()
            except Exception as e:
                print(f"파일 처리 중 오류: {file_path} - {e}")

print(f"\n총 {len(all_songs_data)}개의 곡 정보를 추출했습니다.")

# --- 3. 데이터 저장 ---
# 리스트에 저장된 모든 노래 정보를 Pandas DataFrame으로 변환 후, CSV 파일로 저장합니다.
df_summary = pd.DataFrame(all_songs_data)
df_summary.to_csv(output_csv_file, index=False, encoding='utf-8-sig')

print(f"모든 정보가 '{output_csv_file}' 파일에 저장되었습니다.")