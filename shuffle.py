from datasets import load_dataset
import pandas as pd

# Load dataset from HuggingFace
ds = load_dataset("Hello-SimpleAI/HC3", "all")
df = ds["train"].to_pandas()

# Shuffle rows to randomize sources
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save locally
shuffled_df.to_csv("shuffled_hc3_dataset.csv", index=False)
print(f"Saved shuffled dataset with {len(shuffled_df)} rows.")