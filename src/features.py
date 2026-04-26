import pandas as pd
import numpy as np
from utils import Player, MatchBuilder


def feature_engineering(df : pd.DataFrame) -> pd.DataFrame:
    
    matches = [] #: list[pd.Series] = []
    players : dict[str, Player] = {}

    for _, row in df.iterrows():

        winner_player_name : str = row["Winner"]
        loser_player_name : str = row["Loser"]

        winner_player : Player = players.get(winner_player_name, Player())
        loser_player : Player = players.get(loser_player_name, Player())

        h2h_diff = winner_player.wins[loser_player_name] - loser_player.wins[winner_player_name]
        win_rate_diff = winner_player.get_win_rate() - loser_player.get_win_rate()
        fatigue_diff = winner_player.get_fatigue(row["Date"]) - loser_player.get_fatigue(row["Date"])
        last_k_matches_win_rate_diff = winner_player.get_last_k_matches_win_rate() - loser_player.get_last_k_matches_win_rate()

        match_builder = MatchBuilder()

        if np.random.rand() < 0.5:
            
            matches.append(
                match_builder
                .add_first_player(winner_player_name)
                .add_second_player(loser_player_name)
                .add_rank_diff(row["WRank"] - row["LRank"])
                .add_h2h_diff(h2h_diff)
                .add_win_rate_diff(win_rate_diff)
                .add_max_bet_diff(row["MaxW"] - row["MaxL"])
                .add_avg_bet_diff(row["AvgW"] - row["AvgL"])
                .add_fatigue_diff(fatigue_diff)
                .add_last_k_matches_win_rate_diff(last_k_matches_win_rate_diff)
                .add_winner(0)
                .build()
            )
        
        else:

            matches.append(
                match_builder
                .add_first_player(loser_player_name)
                .add_second_player(winner_player_name)
                .add_rank_diff(row["LRank"] - row["WRank"])
                .add_h2h_diff(-1 * h2h_diff)
                .add_win_rate_diff(-1 * win_rate_diff)
                .add_max_bet_diff(row["MaxL"] - row["MaxW"])
                .add_avg_bet_diff(row["AvgL"] - row["AvgW"])
                .add_fatigue_diff(-1 * fatigue_diff)
                .add_last_k_matches_win_rate_diff(-1 * last_k_matches_win_rate_diff)
                .add_winner(1)
                .build()
            )

        winner_player.game_update(
            win = True,
            rival = loser_player_name,
            match_date = row["Date"]
        )
        loser_player.game_update(
            win = False,
            rival = winner_player_name,
            match_date = row["Date"]
        )
    
    return pd.DataFrame(matches)
