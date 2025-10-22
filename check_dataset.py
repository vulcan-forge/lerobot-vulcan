import pandas as pd

# Check data
df = pd.read_parquet('so100-001__green-tape-in-cup-a001-chris/data/chunk-000/file-000.parquet')
print(f'Total frames: {len(df)}')
print(f'Unique episodes: {sorted(df["episode_index"].unique().tolist())}')
print('\nFrames per episode:')
print(df.groupby('episode_index').size())

# Check metadata
meta = pd.read_parquet('so100-001__green-tape-in-cup-a001-chris/meta/episodes/chunk-000/file-000.parquet')
print(f'\nMetadata episodes: {len(meta)}')
print(meta)


