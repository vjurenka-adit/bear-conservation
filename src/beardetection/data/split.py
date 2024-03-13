import random

import pandas as pd


def random_split(
    X: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.5,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Returns a dataframe with the following extra columns from X:
    - split: str in {train, val, test}
    """
    assert 0.0 <= train_ratio <= 1.0
    assert 0.0 <= val_ratio <= 1.0
    n = len(X)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * (n - train_size))
    random.Random(random_seed).shuffle(X)
    X_train = X[:train_size]
    X_val = X[train_size : train_size + val_size]
    X_test = X[train_size + val_size :]
    result = []

    for x in X_train:
        result.append({**x, "split": "train"})
    for x in X_val:
        result.append({**x, "split": "val"})
    for x in X_test:
        result.append({**x, "split": "test"})
    return pd.DataFrame(result)


def to_indices(pd_group_items: list[tuple]) -> pd.Index:
    """Collect the pandas group indices from the group items."""
    indices = []
    for _, idx in pd_group_items:
        indices.extend(list(idx))
    return pd.Index(indices, dtype="int64")


def split_by_camera_and_date(
    X: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.5,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Returns a dataframe with the following extra columns from X:
    - split: str in {train, val, test}
    """
    assert 0.0 <= train_ratio <= 1.0
    assert 0.0 <= val_ratio <= 1.0
    df = pd.DataFrame(X)
    group_by_key = ["year", "month", "day", "camera_id"]
    g = df.groupby(group_by_key)
    G = list(g.groups.items())
    random.Random(random_seed).shuffle(G)
    n = len(G)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * (n - train_size))
    G_train = G[:train_size]
    G_val = G[train_size : train_size + val_size]
    G_test = G[train_size + val_size :]

    df_train = df.iloc[to_indices(G_train)].copy()
    df_val = df.iloc[to_indices(G_val)].copy()
    df_test = df.iloc[to_indices(G_test)].copy()
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    return pd.concat([df_train, df_val, df_test])
