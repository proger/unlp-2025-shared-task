from pathlib import Path
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Read a Parquet file.")
parser.add_argument("file_path", type=str, help="Path to the Parquet file")
parser.add_argument("output_path", type=str, help="Path to save the JSONL file")
parser.add_argument("--folds", type=str, help="Path prefix to save the folds")
args = parser.parse_args()

df = pd.read_parquet(args.file_path)
df.to_json(args.output_path, orient="records", lines=True, force_ascii=False)

def split_into_folds(data, output_prefix, seed=42):
    # Shuffle and create fold indices
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    fold_sizes = len(data) // 5
    folds = []

    for i in range(5):
        start = i * fold_sizes
        end = start + fold_sizes if i < 4 else len(data)
        folds.append(shuffled_indices[start:end])

    # Save each fold as a separate JSONL file
    for i, fold_indices in enumerate(folds):
        fold_data = data.iloc[fold_indices]
        output_file = f"{output_prefix}_{i + 1}.jsonl"
        fold_data.to_json(output_file, orient="records", lines=True, force_ascii=False)
        print(f"{output_file}")

if args.folds:
    split_into_folds(df, args.folds)
