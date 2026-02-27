from pathlib import Path
import pandas as pd


def main():
    root = Path("data/raw/competition")
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        p = root / name
        print("\n===", name, "===")
        if not p.exists():
            print("MISSING:", p.resolve())
            continue
        df = pd.read_csv(p)
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print(df.head(2).to_string(index=False))


if __name__ == "__main__":
    main()
