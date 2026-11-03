import pandas as pd 
df = pd.read_parquet('data/papers_2015_2025.parquet', engine='pyarrow')
print(df.head())
print(df.columns)