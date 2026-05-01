import argparse
import logging

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from utils import Player, MatchBuilder, Match, get_df_from_pipeline
from data_loader import PROCESSED_DATA_PATH

class FeatureEngineringTransformer:

    def __init__(self) -> None:
        pass

    def fit(self, X : pd.DataFrame, y = None):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:

        df = X.copy()

        TOURNAMENT_COLS = [col for col in df.columns if col.startswith("Tournament")]
        
        matches : list[Match] = [] 
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
            win_streak_diff = winner_player.win_streak - loser_player.win_streak
            lose_streak_diff = winner_player.lose_streak - loser_player.lose_streak

            winner_court_win_rate, winner_surface_win_rate = winner_player.get_court_surface_performance(row["Court"], row["Surface"])
            loser_court_win_rate, loser_surface_win_rate = loser_player.get_court_surface_performance(row["Court"], row["Surface"])
            court_win_rate_diff : float = winner_court_win_rate - loser_court_win_rate
            surface_win_rate_diff : float = winner_surface_win_rate - loser_surface_win_rate

            winner_games_won = row["W1"] + row["W2"] + row["W3"] + row["W4"] + row["W5"]
            loser_games_won = row["L1"] + row["L2"] + row["L3"] + row["L4"] + row["L5"]
            games_played = winner_games_won + loser_games_won

            match_builder = MatchBuilder().add_round(row["Round"]).add_series(row["Series"]).add_date(row["Date"])

            if np.random.rand() < 0.5:
                
                matches.append(
                    match_builder
                    .add_rank_diff(row["WRank"] - row["LRank"])
                    .add_h2h_diff(h2h_diff)
                    .add_date(row["Date"])
                    .add_sets_h2h_diff(sets_h2h_diff)
                    .add_win_rate_diff(win_rate_diff)
                    .add_max_bet_diff(row["MaxW"] - row["MaxL"])
                    .add_avg_bet_diff(row["AvgW"] - row["AvgL"])
                    .add_fatigue_diff(fatigue_diff)
                    .add_last_k_matches_win_rate_diff(last_k_matches_win_rate_diff)
                    .add_court_win_rate_diff(court_win_rate_diff)
                    .add_surface_win_rate_diff(surface_win_rate_diff)
                    .add_win_streak_diff(win_streak_diff)
                    .add_lose_streak_diff(lose_streak_diff)
                    .add_winner(0)
                    .build()
                )
            
            else:

                matches.append(
                    match_builder
                    .add_rank_diff(row["LRank"] - row["WRank"])
                    .add_h2h_diff(-1 * h2h_diff)
                    .add_date(row["Date"])
                    .add_sets_h2h_diff(-1 * sets_h2h_diff)
                    .add_win_rate_diff(-1 * win_rate_diff)
                    .add_max_bet_diff(row["MaxL"] - row["MaxW"])
                    .add_avg_bet_diff(row["AvgL"] - row["AvgW"])
                    .add_fatigue_diff(-1 * fatigue_diff)
                    .add_last_k_matches_win_rate_diff(-1 * last_k_matches_win_rate_diff)
                    .add_court_win_rate_diff(-1 * court_win_rate_diff)
                    .add_surface_win_rate_diff(-1 * surface_win_rate_diff)
                    .add_win_streak_diff(-1 * win_streak_diff)
                    .add_lose_streak_diff(-1 * lose_streak_diff)
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
        
        matches_df = pd.DataFrame(matches)
        df = pd.concat([df[TOURNAMENT_COLS].reset_index(drop=True), matches_df.reset_index(drop=True)], axis=1)
        return df


def feature_engineering_pipeline(df : pd.DataFrame) -> Pipeline:

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineringTransformer())
        ]
    )

def save_features(df : pd.DataFrame):
    df.to_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx', index=False)

def read_clean_data() -> pd.DataFrame:
    return pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_clean.xlsx')

def create_features(save_data : bool = False) -> pd.DataFrame:
    
    logger.info("Starting feature engineering pipeline")

    df : pd.DataFrame = read_clean_data()
    pipeline : Pipeline = feature_engineering_pipeline(df)
    df = get_df_from_pipeline(pipeline, df)
    if save_data:
        save_features(df)

    logger.info("Feature engineering pipeline completed")

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save preprocessed data to Excel file"
    )

    args = parser.parse_args()

    create_features(save_data=args.save)
