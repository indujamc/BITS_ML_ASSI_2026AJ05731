"""Download the UCI Dry Bean dataset and save it as dry_bean.csv."""
from ucimlrepo import fetch_ucirepo

def main():
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    # UCI target column is named Class.
    y.columns = ["Class"]
    import pandas as pd
    data = pd.concat([X, y], axis=1)
    data.to_csv("dry_bean.csv", index=False)
    print(f"Saved dry_bean.csv: {data.shape[0]} rows, {data.shape[1]-1} features + target")
    print(data["Class"].value_counts())

if __name__ == "__main__":
    main()
