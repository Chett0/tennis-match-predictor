import pandas as pd
import numpy as np
from utils import PROCESSED_DATA_PATH, Player, MatchBuilder


def feature_engineering(df : pd.DataFrame) -> pd.DataFrame:
    
    matches = [] #: list[pd.Series] = []
    players : dict[str, Player] = {}

    for _, row in df.iterrows():

        winner_player_name : str = row["Winner"]
        loser_player_name : str = row["Loser"]

        if winner_player_name not in players:
            players[winner_player_name] = Player()
        if loser_player_name not in players:
            players[loser_player_name] = Player()
        winner_player : Player = players[winner_player_name]
        loser_player : Player = players[loser_player_name]

        h2h_diff = winner_player.wins[loser_player_name]["matches"] - loser_player.wins[winner_player_name]["matches"]
        sets_h2h_diff = winner_player.wins[loser_player_name]["sets"] - loser_player.wins[winner_player_name]["sets"]
        win_rate_diff = winner_player.get_win_rate() - loser_player.get_win_rate()
        max_games = 39 if row["Best of"] == 3 else 65
        fatigue_diff = winner_player.get_fatigue(row["Date"], long_stop_days=15, max_games=max_games) - loser_player.get_fatigue(row["Date"], long_stop_days=15, max_games=max_games)
        last_k_matches_win_rate_diff = winner_player.get_last_k_matches_win_rate() - loser_player.get_last_k_matches_win_rate()

        winner_court_win_rate, winner_surface_win_rate = winner_player.get_court_surface_performance(row["Court"], row["Surface"])
        loser_court_win_rate, loser_surface_win_rate = loser_player.get_court_surface_performance(row["Court"], row["Surface"])
        court_win_rate_diff : float = winner_court_win_rate - loser_court_win_rate
        surface_win_rate_diff : float = winner_surface_win_rate - loser_surface_win_rate

        winner_games_won = row["W1"] + row["W2"] + row["W3"] + row["W4"] + row["W5"]
        loser_games_won = row["L1"] + row["L2"] + row["L3"] + row["L4"] + row["L5"]
        games_played = winner_games_won + loser_games_won

        match_builder = MatchBuilder().add_tournament(row["Tournament"]).add_round(row["Round"]).add_series(row["Series"]).add_date(row["Date"])

        if np.random.rand() < 0.5:
            
            matches.append(
                match_builder
                .add_rank_diff(row["WRank"] - row["LRank"])
                .add_h2h_diff(h2h_diff)
                .add_sets_h2h_diff(sets_h2h_diff)
                .add_win_rate_diff(win_rate_diff)
                .add_max_bet_diff(row["MaxW"] - row["MaxL"])
                .add_avg_bet_diff(row["AvgW"] - row["AvgL"])
                .add_fatigue_diff(fatigue_diff)
                .add_last_k_matches_win_rate_diff(last_k_matches_win_rate_diff)
                .add_court_win_rate_diff(court_win_rate_diff)
                .add_surface_win_rate_diff(surface_win_rate_diff)
                .add_winner(0)
                .build()
            )
        
        else:

            matches.append(
                match_builder
                .add_rank_diff(row["LRank"] - row["WRank"])
                .add_h2h_diff(-1 * h2h_diff)
                .add_sets_h2h_diff(-1 * sets_h2h_diff)
                .add_win_rate_diff(-1 * win_rate_diff)
                .add_max_bet_diff(row["MaxL"] - row["MaxW"])
                .add_avg_bet_diff(row["AvgL"] - row["AvgW"])
                .add_fatigue_diff(-1 * fatigue_diff)
                .add_last_k_matches_win_rate_diff(-1 * last_k_matches_win_rate_diff)
                .add_court_win_rate_diff(-1 * court_win_rate_diff)
                .add_surface_win_rate_diff(-1 * surface_win_rate_diff)
                .add_winner(1)
                .build()
            )

        winner_player.game_update(
            win = True,
            rival = loser_player_name,
            match_date = row["Date"],
            court = row["Court"],
            surface = row["Surface"],
            sets_won = row["Wsets"],
            games_played = games_played,
        )
        loser_player.game_update(
            win = False,
            rival = winner_player_name,
            match_date = row["Date"],
            court = row["Court"],
            surface = row["Surface"],
            sets_won = row["Lsets"],
            games_played = games_played,
        )
    
    return pd.DataFrame(matches)

def save_features(df : pd.DataFrame):
    df.to_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx', index=False)

def feature_engineering_pipeline():
    df = pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_clean.xlsx')
    df = feature_engineering(df)
    save_features(df)

if __name__ == "__main__":
    feature_engineering_pipeline()