import pandas as pd
import os

folder = "data/raw/opentargets/"
file = os.listdir(folder)[0]

df = pd.read_parquet(os.path.join(folder, file))

print(df.shape)
print(df.columns)
print(df.head())