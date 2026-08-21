import copy
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.match import Match, MatchBuilder
from src.utils.player import Player

class FeatureEngineringTransformer(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        pass

    def fit(self, X : pd.DataFrame, y = None):
        # cols of ohe of tournament
        self.tournament_cols_ = [col for col in X.columns if col.startswith("Tournament")]
        self.players_ : dict[str, Player] = {}
        self.matches_ : list[Match] = []
        self.create_features(X, self.players_, self.matches_)
        self.n_training_matches_ = len(self.matches_)
        self.train_index_ = X.index
        return self


    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        if X.index.equals(self.train_index_):
            matches_df = pd.DataFrame(self.matches_)
        else:
            players_copy = copy.deepcopy(self.players_)
            matches_copy = self.matches_.copy()
            matches = self.create_features(X, players_copy, matches_copy)
            matches_df = pd.DataFrame(matches[self.n_training_matches_:])

        X = pd.concat([X[self.tournament_cols_].reset_index(drop=True), matches_df.reset_index(drop=True)], axis=1)
        return X

    
    def create_features(
            self, 
            df : pd.DataFrame, 
            players : dict[str, Player], 
            matches : list[Match]
    ) -> list[Match]:
        
        for _, row in df.iterrows():
            player1_name : str = row["player1"]
            player2_name : str = row["player2"]

            if player1_name not in players:
                players[player1_name] = Player()
            if player2_name not in players:
                players[player2_name] = Player()
            player1 : Player = players[player1_name]
            player2 : Player = players[player2_name]

            rank_diff = row["player1_Rank"] - row["player2_Rank"]
            h2h_diff = player1.wins[player2_name]["matches"] - player2.wins[player1_name]["matches"]
            sets_h2h_diff = player1.wins[player2_name]["sets"] - player2.wins[player1_name]["sets"]
            win_rate_diff = player1.get_win_rate() - player2.get_win_rate()

            max_games = 39 if row["Best of"] == 3 else 65
            fatigue_diff = player1.get_fatigue(row["Date"], long_stop_days=15, max_games=max_games) - player2.get_fatigue(row["Date"], long_stop_days=15, max_games=max_games)
            last_k_matches_win_rate_diff = player1.get_last_k_matches_win_rate() - player2.get_last_k_matches_win_rate()
            win_streak_diff = player1.win_streak - player2.win_streak
            lose_streak_diff = player1.lose_streak - player2.lose_streak

            court_win_rate_diff : float = player1.get_court_performance(row["Court"]) - player2.get_court_performance(row["Court"])
            surface_win_rate_diff : float = player1.get_surface_performance(row["Surface"]) - player2.get_surface_performance(row["Surface"])

            player1_games_won = row["player1_1"] + row["player1_2"] + row["player1_3"] + row["player1_4"] + row["player1_5"]
            player2_games_won = row["player2_1"] + row["player2_2"] + row["player2_3"] + row["player2_4"] + row["player2_5"]
            games_played = player1_games_won + player2_games_won

            match_builder = MatchBuilder() \
                                .add_round(row["Round"]) \
                                .add_series(row["Series"]) \
                                .add_date(row["Date"]) 

            matches.append(
                match_builder
                .add_rank_diff(rank_diff)
                .add_h2h_diff(h2h_diff)
                .add_date(row["Date"])
                .add_sets_h2h_diff(sets_h2h_diff)
                .add_win_rate_diff(win_rate_diff)
                .add_max_bet_diff(row["player1_Max"] - row["player2_Max"])
                .add_avg_bet_diff(row["player1_Avg"] - row["player2_Avg"])
                .add_fatigue_diff(fatigue_diff)
                .add_last_k_matches_win_rate_diff(last_k_matches_win_rate_diff)
                .add_court_win_rate_diff(court_win_rate_diff)
                .add_surface_win_rate_diff(surface_win_rate_diff)
                .add_win_streak_diff(win_streak_diff)
                .add_lose_streak_diff(lose_streak_diff)
                .build()
            )
            
            player1_won : bool = player1_name == row["match_winner"]
            player1.game_update(
                win = player1_won,
                rival = player2_name,
                match_date = row["Date"],
                court = row["Court"],
                surface = row["Surface"],
                sets_won = row["player1_sets"],
                games_played = games_played,
            )
            player2.game_update(
                win = not player1_won,
                rival = player1_name,
                match_date = row["Date"],
                court = row["Court"],
                surface = row["Surface"],
                sets_won = row["player2_sets"],
                games_played = games_played,
            )

        return matches