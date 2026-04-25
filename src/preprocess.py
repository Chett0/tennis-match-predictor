import pandas as pd
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

    return pd.concat(dfs, ignore_index=True)