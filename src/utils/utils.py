import pandas as pd
from sklearn.pipeline import Pipeline

GAME_COLS = [
    "player1_1", 
    "player2_1", 
    "player1_2", 
    "player2_2", 
    "player1_3", 
    "player2_3", 
    "player1_4", 
    "player2_4", 
    "player1_5", 
    "player2_5"
]

POINT_COLS = ["player1_Pts", "player2_Pts"]
RANK_COLS = ["player1_Rank", "player2_Rank"]
SETS_COLS = ["player1_sets", "player2_sets"]
CAT_COLS = ["Tournament"]

BET_COLS = [
    "player1_B365", 
    "player2_B365", 
    "player1_PS", 
    "player2_PS", 
    "player1_Max", 
    "player2_Max", 
    "player1_Avg", 
    "player2_Avg"
]


ROUND_ORDER = [[
    "Round Robin",
    "1st Round",
    "2nd Round",
    "3rd Round",
    "4th Round",
    "Quarterfinals",
    "Semifinals",
    "The Final"
]]

SERIES_ORDER = [[
    "ATP250",
    "ATP500",
    "Masters 1000",
    "Masters Cup",
    "Grand Slam"
]]

swap_cols = [
    ["WRank", "LRank"],
    ["WPts", "LPts"],
    ["B365W", "B365L"],
    ["BFEW", "BFEL"],
    ["PSW", "PSL"],
    ["MaxW", "MaxL"],
    ["AvgW", "AvgL"],
    ["W1", "L1"],
    ["W2", "L2"],
    ["W3", "L3"],
    ["W4", "L4"],
    ["W5", "L5"],
    ["Wsets", "Lsets"]
]

def get_df_from_pipeline(pipeline : Pipeline, df : pd.DataFrame):
    df_ = pipeline.fit_transform(df)
    return pd.DataFrame(df_)


