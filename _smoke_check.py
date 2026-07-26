import pandas as pd
from src import validation as v

df = pd.DataFrame({
    "pickup_datetime": pd.to_datetime(["2026-04-01", "2026-04-05", None]),
    "PULocationID": [1, 2, 2],
})
df = pd.concat([df, df.iloc[[1]]], ignore_index=True)  # inject a duplicate

print("schema:", v.profile_schema(df))
print("nulls:", v.profile_nulls(df))
print("range:", v.pickup_date_range(df))
print("dups:", v.count_duplicates(df))

# empty frame edge case
empty = pd.DataFrame({"pickup_datetime": pd.to_datetime([])})
print("empty nulls:", v.profile_nulls(empty))

try:
    v.load_parquet("data/does_not_exist.parquet")
except FileNotFoundError as e:
    print("OK missing:", str(e)[:70])
