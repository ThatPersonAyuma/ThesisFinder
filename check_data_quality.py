import pandas as pd

# Load main dataset
df = pd.read_parquet('data/completed.parquet')

print("="*60)
print("DATA QUALITY CHECK - completed.parquet")
print("="*60)
print(f"Total rows: {len(df):,}")
print()

# Check all relevant fields
fields = ['year', 'primary_category', 'authors', 'categories', 'keywords']
print("NULL COUNT BY FIELD:")
print("-"*60)
for field in fields:
    null_count = df[field].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"{field:20s}: {null_count:7,} nulls ({null_pct:5.2f}%)")

print()
print("-"*60)

# Check authors field details
print("\nAUTHORS FIELD ANALYSIS:")
print("-"*60)
authors_sample = df[df['authors'].notna()]['authors'].head(5)
print("Sample values:")
for i, val in enumerate(authors_sample, 1):
    print(f"  {i}. {val[:100]}...")

# Check categories field details
print("\nCATEGORIES FIELD ANALYSIS:")
print("-"*60)
categories_sample = df[df['categories'].notna()]['categories'].head(5)
print("Sample values:")
for i, val in enumerate(categories_sample, 1):
    print(f"  {i}. {val}")

# Check primary_category distribution
print("\nPRIMARY_CATEGORY DISTRIBUTION (Top 10):")
print("-"*60)
print(df['primary_category'].value_counts().head(10))

# Check year distribution
print("\nYEAR DISTRIBUTION:")
print("-"*60)
print(df['year'].value_counts().sort_index())

print()
print("="*60)
print("Checking secondary dataset...")
print("="*60)

# Load secondary dataset
try:
    df_keywords = pd.read_parquet('data/papers_2015_2025_keywords.parquet')
    print(f"Total rows: {len(df_keywords):,}")
    print(f"Columns: {list(df_keywords.columns)}")
    print()
    
    if 'keywords' in df_keywords.columns:
        keywords_null = df_keywords['keywords'].isnull().sum()
        keywords_pct = (keywords_null / len(df_keywords)) * 100
        print(f"Keywords null count: {keywords_null:,} ({keywords_pct:.2f}%)")
        
        print("\nKeywords sample (non-null):")
        keywords_sample = df_keywords[df_keywords['keywords'].notna()]['keywords'].head(5)
        for i, val in enumerate(keywords_sample, 1):
            print(f"  {i}. {val}")
    
    # Check if IDs match
    if 'id' in df_keywords.columns:
        print(f"\nID overlap check:")
        overlap = set(df['id']) & set(df_keywords['id'])
        print(f"  Main dataset IDs: {len(df):,}")
        print(f"  Keywords dataset IDs: {len(df_keywords):,}")
        print(f"  Overlapping IDs: {len(overlap):,}")
        print(f"  Can be merged: {'YES' if len(overlap) > 0 else 'NO'}")
        
except Exception as e:
    print(f"Error loading secondary dataset: {e}")

print()
print("="*60)
print("RECOMMENDATIONS:")
print("="*60)
