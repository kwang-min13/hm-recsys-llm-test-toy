import duckdb

con = duckdb.connect(':memory:')

# Check user_features columns
print("User Features Columns:")
cols = con.execute("DESCRIBE SELECT * FROM read_parquet('data/features/user_features.parquet')").fetchall()
for col in cols:
    print(f"  {col[0]}: {col[1]}")

print("\nSample data:")
sample = con.execute("SELECT * FROM read_parquet('data/features/user_features.parquet') LIMIT 3").fetchall()
for row in sample:
    print(f"  {row}")

con.close()
