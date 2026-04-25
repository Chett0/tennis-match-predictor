import pandas as pd
import numpy as np
import glob

def load_2025_data() -> pd.DataFrame:
    data : pd.DataFrame = pd.read_excel('../data/2025.xlsx')
    return data

def load_data() -> pd.DataFrame:
    files = glob.glob('../data/*.xlsx')

    dfs : list[pd.DataFrame] = []

    for file in files:
        df : pd.DataFrame = pd.read_excel(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_data(df : pd.DataFrame) -> pd.DataFrame:

    df.drop(columns=["BFEW", "BFEL"], inplace=True, errors="ignore")
    df = df[df["Comment"] != "Walkover"]
    
    bets = [
        "B365W",
        "B365L",
        "PSW",
        "PSL",
        "MaxW",
        "MaxL",
        "AvgW",
        "AvgL"
    ]

    df[bets] = df[bets].fillna(0.5)

    max_rank = max(df["WRank"].max(), df["LRank"].max())
    df[["WRank", "LRank"]] = df[["WRank", "LRank"]].fillna(max_rank + 100)

    games = [
        "W1",
        "L1",
        "W2",
        "L2",
        "W3",
        "L3",
        "W4",
        "L4",
        "W5",
        "L5"
    ]
    df[games] = df[games].fillna(0)

    mask = df["Best of"].isna()

    condizione = (
        (df["W4"] != 0) | (df["W5"] != 0) |
        (
            (df["W1"] > df["L1"]) &
            (df["W2"] > df["L2"]) &
            (df["W3"] > df["L3"])
        )
    )

    df.loc[mask, "Best of"] = np.where(
        condizione.loc[mask],
        5,
        3
    )

    return df


raw_data = load_data()
raw_data = clean_data(raw_data)
raw_data.info()