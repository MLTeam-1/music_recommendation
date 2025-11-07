import pandas as pd

# 사용자 환경에 맞게 원본 tsv 파일 경로를 지정하세요
input_tsv = '../data/userid-timestamp-artid-artname-traid-traname.tsv'  # 파일 위치와 이름 확인 필수

# 정확한 컬럼명
cols = ['userid', 'timestamp', 'artid', 'artname', 'traid', 'traname']

# 데이터 전체 읽기 (큰 파일일 경우 메모리 고려 필요)
df = pd.read_csv(input_tsv, sep='\t', names=cols, header=None, engine='python', on_bad_lines='skip')

# traid 결측 비율 확인
missing_traid_ratio = df['traid'].isna().mean()

# traid 포함하여 그룹화하고, 없는 경우 따로 그룹화
df_with_traid = df.groupby(['userid', 'artid', 'artname', 'traid', 'traname']).size().reset_index(name='play_count')
df_without_traid = df.groupby(['userid', 'artid', 'artname', 'traname']).size().reset_index(name='play_count')

print(f'Total rows in raw data: {len(df)}')
print(f'Missing traid ratio: {missing_traid_ratio:.4f}')
print(f'Number of groups with traid: {len(df_with_traid)}')
print(f'Number of groups without traid: {len(df_without_traid)}')

# traname별 고유 traid 개수 샘플 보기
trackid_counts = df.groupby('traname')['traid'].nunique()
print('Sample traname unique traid counts:')
print(trackid_counts.head(10))
