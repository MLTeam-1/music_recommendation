# ======================================================================================
# [프로젝트] 이기종 음악 데이터셋을 활용한 개인화 추천 시스템 구축
# [스크립트 목적] Million Song Dataset(MSD) 원본 데이터 처리
# [End-to-End 단계] 2. 데이터 수집(Sourcing) 및 3. 데이터 정제/전처리(Preprocessing)
#
# [설명]
# 이 스크립트는 Million Song Dataset의 파편화된 수십만 개의 .h5 파일들을 순회하며,
# 각 노래의 핵심 메타데이터(ID, 아티스트, 제목 등)를 추출합니다.
# 이 과정을 통해, 비구조적인 원본 데이터를 분석 가능한 형태인
# 단일 CSV 파일로 변환하는 것을 목표로 합니다.
# 이는 '양(Quantity)' 중심 데이터셋을 구축하기 위한 첫 번째 단계입니다.
# ======================================================================================


# --- 라이브러리 불러오기 ---
import os       # 운영체제와 상호작용하기 위한 라이브러리 (파일 경로, 폴더 순회 등)
import h5py     # HDF5 형식(.h5) 파일을 읽고 쓰기 위한 라이브러리
import pandas as pd # 데이터를 표 형태로 다루기 위한 강력한 라이브러리 (DataFrame)

# --- 설정: 입력 폴더 및 출력 파일 경로 ---

# [단계 2: 데이터 수집]
# Million Song Dataset의 Subset 원본 데이터가 저장된 최상위 폴더 경로를 지정합니다.
# '../'는 현재 위치의 상위 폴더를 의미하며, 이는 코드와 데이터가 분리된 좋은 구조입니다.
msd_subset_path = '../data/MillionSongSubset'

# [단계 3: 데이터 정제/전처리]
# 추출된 메타데이터를 저장할 CSV 파일의 경로와 이름을 지정합니다.
# 이 파일이 이 스크립트의 최종 결과물(output)이 됩니다.
output_csv_file = '../result/msd_metadata_with_mbid.csv'

# --- 데이터 추출을 위한 초기화 ---

# 모든 노래 정보를 임시로 저장할 빈 리스트(list)를 생성합니다.
# 딕셔너리 형태로 각 노래의 정보를 담은 뒤, 마지막에 DataFrame으로 한 번에 변환하는 것이 효율적입니다.
all_songs_data = []
# 처리한 파일의 개수를 세어 진행 상황을 파악하기 위한 카운터 변수입니다.
file_count = 0

# --- 핵심 로직: 모든 .h5 파일 순회 및 데이터 추출 ---

print("--- [단계 3: 데이터 정제/전처리] MSD .h5 파일 순회 및 메타데이터 추출 시작 ---")
# os.walk() 함수는 지정된 폴더(msd_subset_path)와 그 안의 모든 하위 폴더를 재귀적으로 탐색합니다.
# root: 현재 탐색 중인 폴더 경로
# dirs: 현재 폴더 안에 있는 하위 폴더들의 목록
# files: 현재 폴더 안에 있는 파일들의 목록
for root, dirs, files in os.walk(msd_subset_path):
    # 현재 폴더에 있는 파일 목록을 하나씩 순회합니다.
    for file in files:
        # 파일 이름이 '.h5'로 끝나는 HDF5 파일인 경우에만 로직을 실행합니다.
        if file.endswith('.h5'):
            # 현재 폴더 경로(root)와 파일 이름(file)을 합쳐서 운영체제에 맞는 전체 파일 경로를 만듭니다.
            file_path = os.path.join(root, file)
            
            # 파일 처리 중 손상된 파일 등의 예상치 못한 오류가 발생하더라도
            # 전체 프로그램이 멈추지 않도록 예외 처리(try...except) 구문을 사용합니다.
            try:
                # h5py.File()을 사용하여 .h5 파일을 '읽기 모드(r)'로 엽니다.
                # 'with' 구문을 사용하면 파일 작업이 끝났을 때 자동으로 파일을 닫아주므로 메모리 누수를 방지합니다.
                with h5py.File(file_path, 'r') as f:
                    # HDF5 파일 내에서 'metadata'라는 그룹(폴더와 유사)을 가져옵니다.
                    meta = f.get('metadata', None)
                    
                    # 만약 'metadata' 그룹이 존재하지 않으면, 유효한 데이터가 아니므로
                    # 이 파일은 건너뛰고(continue) 다음 파일 처리를 시작합니다.
                    if meta is None:
                        continue

                    # HDF5 파일 내의 문자열은 종종 바이트(bytes) 형식으로 저장되어 있습니다.
                    # 이를 파이썬의 일반 문자열(string)로 안전하게 변환해주는 헬퍼(helper) 함수를 정의합니다.
                    def safe_decode(value):
                        # 만약 입력값이 바이트(bytes) 타입이면,
                        if isinstance(value, bytes):
                            # 'utf-8' 인코딩 형식으로 디코딩(문자열로 변환)하여 반환합니다.
                            return value.decode('utf-8')
                        # 바이트 타입이 아니면 (예: 이미 문자열이거나 숫자), 원래 값을 그대로 반환합니다.
                        return value

                    # 'metadata' 그룹에서 필요한 데이터 필드들을 안전하게 읽어옵니다.
                    # .get(키, 기본값)을 사용하여, 해당 키가 없을 경우 오류 대신 지정된 기본값을 반환하도록 합니다.
                    song_id = safe_decode(meta.get('song_id', b''))
                    artist_name = safe_decode(meta.get('artist_name', b''))
                    title = safe_decode(meta.get('title', b''))
                    # year는 보통 정수형이므로, 없을 경우 -1을 기본값으로 설정하여 결측치를 표시합니다.
                    year = meta.get('year', -1)

                    # MusicBrainz ID는 없을 수도 있으므로, 바이트 문자열을 가져온 뒤 값이 있는지 확인하고 변환합니다.
                    mbid_bytes = meta.get('musicbrainz_trackid', b'')
                    mbid = safe_decode(mbid_bytes) if mbid_bytes else '' # mbid_bytes가 비어있지 않은 경우에만 디코딩

                    # 추출한 데이터들을 딕셔너리(dictionary) 형태로 정리하여 가독성을 높입니다.
                    # 이 딕셔너리가 나중에 CSV 파일의 한 행(row)이 됩니다.
                    song_data = {
                        'song_id': song_id,
                        'musicbrainz_trackid': mbid,
                        'artist_name': artist_name,
                        'title': title,
                        'year': year,
                    }
                    # 정리된 노래 정보를 맨 처음에 만들었던 리스트(all_songs_data)에 추가(append)합니다.
                    all_songs_data.append(song_data)

                    # 처리한 파일 카운터를 1 증가시킵니다.
                    file_count += 1
                    # 1000개의 파일을 처리할 때마다 진행 상황을 화면에 출력하여,
                    # 프로그램이 멈추지 않고 잘 동작하고 있음을 사용자에게 알려줍니다.
                    if file_count % 1000 == 0:
                        print(f"{file_count}개 파일 처리 완료...")

            # try 블록 안에서 오류(Exception)가 발생하면 이 부분을 실행합니다.
            except Exception as e:
                # 어떤 파일에서 어떤 오류가 발생했는지 구체적인 정보를 화면에 출력하여 디버깅을 돕습니다.
                print(f"파일 처리 중 오류: {file_path} - {e}")

# 모든 파일 순회가 끝난 후, 최종적으로 추출된 노래의 총 개수를 출력하여 결과를 요약합니다.
print(f"\n총 {len(all_songs_data)}곡 데이터 추출 완료.")

# --- 데이터 저장 ---

# 지금까지 리스트에 담아온 모든 딕셔너리 데이터들을 Pandas의 DataFrame(표) 형태로 한 번에 변환합니다.
# 이 방식이 리스트에 한 줄씩 추가하는 것보다 훨씬 빠릅니다.
df = pd.DataFrame(all_songs_data)

# 생성된 DataFrame을 최종 결과물인 CSV 파일로 저장합니다.
# index=False: DataFrame의 인덱스(0, 1, 2, ...)가 파일에 별도의 열로 저장되는 것을 방지합니다.
# encoding='utf-8-sig': 한글이나 특수문자가 포함된 경우, Microsoft Excel에서 파일을 열 때 글자가 깨지는 현상을 방지합니다.
df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')

# 모든 작업이 성공적으로 완료되었음을 알리는 최종 메시지를 출력합니다.
print(f"결과가 '{output_csv_file}'에 저장되었습니다.")