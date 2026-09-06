import copy
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.match import Match, MatchBuilder
from src.utils.player import Player, WeightedRankingMethod, WeightedPointsMethod

class FeatureEngineringTransformer(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            h2h_decay_days : float = 365,
            weighted_ranking_method : WeightedRankingMethod = WeightedRankingMethod.INVSQRT,
            weighted_points_method : WeightedPointsMethod = WeightedPointsMethod.CUBE
    ) -> None:
        self._h2h_decay_days = h2h_decay_days
        self.weighted_ranking_method = weighted_ranking_method
        self.weighted_points_method = weighted_points_method

    @property
    def h2h_decay_days(self):
        """Field to control decay of head-to-head matches. The default is 365 days."""
        return self._h2h_decay_days

    @h2h_decay_days.setter
    def h2h_decay_days(self, value):
        if value <= 0:
            raise ValueError("h2h_decay_days must be positive")
        self._h2h_decay_days = value

    # @property
    # def win_ratio_decay_days(self):
    #     """Field to control recency weighting of win ratios. The default is 365 days."""
    #     return self._win_ratio_decay_days

    # @win_ratio_decay_days.setter
    # def win_ratio_decay_days(self, value):
    #     if value <= 0:
    #         raise ValueError("win_ratio_decay_days must be positive")
    #     self._win_ratio_decay_days = value


    def fit(self, X : pd.DataFrame, y = None):
        # ohe cols of tournament, surface, court
        self.tournament_cols_ = [col for col in X.columns if col.startswith("Tournament_")]
        self.surface_cols_ = [col for col in X.columns if col.startswith("Surface_")]
        self.court_cols_ = [col for col in X.columns if col.startswith("Court_")]

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
            # deep copy required to avoid modifying the fitted players and matches when transforming new data
            players_copy = copy.deepcopy(self.players_)
            matches_copy = self.matches_.copy()
            matches = self.create_features(X, players_copy, matches_copy)
            matches_df = pd.DataFrame(matches[self.n_training_matches_:])

        X = pd.concat([
            X[self.tournament_cols_].reset_index(drop=True), 
            X[self.surface_cols_].reset_index(drop=True), 
            X[self.court_cols_].reset_index(drop=True), 
            matches_df.reset_index(drop=True)], axis=1
        )
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

            pts_diff = row["player1_Pts"] - row["player2_Pts"]
            weighted_pts_diff = (
                player1.get_weighted_points(row["player1_Pts"], self.weighted_points_method)
                - player2.get_weighted_points(row["player2_Pts"], self.weighted_points_method)
            )

            rank_diff = row["player1_Rank"] - row["player2_Rank"]
            weighted_ranking_diff = player1.get_weighted_ranking(row["player1_Rank"], self.weighted_ranking_method) - player2.get_weighted_ranking(row["player2_Rank"], self.weighted_ranking_method)
            
            h2h_diff = player1.wins[player2_name]["matches"] - player2.wins[player1_name]["matches"]
            h2h_weighted_diff = (
                player1.get_h2h_weighted_wins(
                    player2_name, row["Date"], self.h2h_decay_days
                )
                - player2.get_h2h_weighted_wins(
                    player1_name, row["Date"], self.h2h_decay_days
                )
            )
            sets_h2h_diff = player1.wins[player2_name]["sets"] - player2.wins[player1_name]["sets"]
            win_rate_diff = player1.get_win_rate() - player2.get_win_rate()
            win_rate_bestof3_diff = player1.get_win_rate_by_best_of(3) - player2.get_win_rate_by_best_of(3)
            win_rate_bestof5_diff = player1.get_win_rate_by_best_of(5) - player2.get_win_rate_by_best_of(5)
            number_recent_matches_diff = player1.get_matches_played_last_days(row["Date"]) - player2.get_matches_played_last_days(row["Date"])
            last_k_games_win_rate_diff = player1.get_last_k_games_win_rate() - player2.get_last_k_games_win_rate()
            last_k_sets_win_rate_diff = player1.get_last_k_sets_win_rate() - player2.get_last_k_sets_win_rate()

            max_games = 39 if row["Best of"] == 3 else 65
            fatigue_diff = player1.get_fatigue(row["Date"], long_stop_days=15, max_games=max_games) - player2.get_fatigue(row["Date"], long_stop_days=15, max_games=max_games)
            last_k_matches_win_rate_diff = player1.get_last_k_matches_win_rate() - player2.get_last_k_matches_win_rate()
            last_k_matches_rank_variation_diff = player1.get_last_k_matches_rank_variation(row["player1_Rank"]) - player2.get_last_k_matches_rank_variation(row["player2_Rank"])
            last_k_non_completed_matches_diff = player1.get_last_k_non_completed_matches() - player2.get_last_k_non_completed_matches()
            win_streak_diff = player1.win_streak - player2.win_streak
            lose_streak_diff = player1.lose_streak - player2.lose_streak

            court_win_rate_diff : float = player1.get_court_win_rate(row["Court"]) - player2.get_court_win_rate(row["Court"])
            tournament_win_rate_diff : float = player1.get_tournament_win_rate(row["Tournament"]) - player2.get_tournament_win_rate(row["Tournament"])
            surface_win_rate_diff : float = player1.get_surface_win_rate(row["Surface"]) - player2.get_surface_win_rate(row["Surface"])
            round_win_rate_diff : float = player1.get_round_win_rate(row["Round"]) - player2.get_round_win_rate(row["Round"])
            series_win_rate_diff : float = player1.get_series_win_rate(row["Series"]) - player2.get_series_win_rate(row["Series"])
            h2h_court_diff : float = player1.get_h2h_court_wins(row["Court"], player2_name) - player2.get_h2h_court_wins(row["Court"], player1_name)
            h2h_tournament_diff : float = player1.get_h2h_tournament_wins(row["Tournament"], player2_name) - player2.get_h2h_tournament_wins(row["Tournament"], player1_name)
            h2h_surface_diff : float = player1.get_h2h_surface_wins(row["Surface"], player2_name) - player2.get_h2h_surface_wins(row["Surface"], player1_name)
            h2h_round_diff : float = player1.get_h2h_round_wins(row["Round"], player2_name) - player2.get_h2h_round_wins(row["Round"], player1_name)
            h2h_series_diff : float = player1.get_h2h_series_wins(row["Series"], player2_name) - player2.get_h2h_series_wins(row["Series"], player1_name)

            player1_games_won = row["player1_1"] + row["player1_2"] + row["player1_3"] + row["player1_4"] + row["player1_5"]
            player2_games_won = row["player2_1"] + row["player2_2"] + row["player2_3"] + row["player2_4"] + row["player2_5"]
            games_played = player1_games_won + player2_games_won

            matches.append(
                MatchBuilder()
                .add_round(row["Round"]) 
                .add_series(row["Series"]) 
                .add_pts_diff(pts_diff)
                .add_weighted_pts_diff(weighted_pts_diff)
                .add_rank_diff(rank_diff)
                .add_weighted_ranking_diff(weighted_ranking_diff)
                .add_h2h_diff(h2h_diff)
                .add_h2h_weighted_diff(h2h_weighted_diff)
                .add_sets_h2h_diff(sets_h2h_diff)
                .add_win_rate_diff(win_rate_diff)
                .add_win_rate_bestof3_diff(win_rate_bestof3_diff)
                .add_win_rate_bestof5_diff(win_rate_bestof5_diff)
                .add_number_recent_matches_diff(number_recent_matches_diff)
                .add_last_k_games_win_rate_diff(last_k_games_win_rate_diff)
                .add_last_k_sets_win_rate_diff(last_k_sets_win_rate_diff)
                .add_max_bet_diff(row["player1_Max"] - row["player2_Max"])
                .add_avg_bet_diff(row["player1_Avg"] - row["player2_Avg"])
                .add_fatigue_diff(fatigue_diff)
                .add_last_k_matches_win_rate_diff(last_k_matches_win_rate_diff)
                .add_last_k_matches_rank_variation_diff(last_k_matches_rank_variation_diff)
                .add_court_win_rate_diff(court_win_rate_diff)
                .add_tournament_win_rate_diff(tournament_win_rate_diff)
                .add_surface_win_rate_diff(surface_win_rate_diff)
                .add_last_k_non_completed_matches_diff(last_k_non_completed_matches_diff)
                .add_round_win_rate_diff(round_win_rate_diff)
                .add_series_win_rate_diff(series_win_rate_diff)
                .add_h2h_court_diff(h2h_court_diff)
                .add_h2h_tournament_diff(h2h_tournament_diff)
                .add_h2h_surface_diff(h2h_surface_diff)
                .add_h2h_round_diff(h2h_round_diff)
                .add_h2h_series_diff(h2h_series_diff)
                .add_win_streak_diff(win_streak_diff)
                .add_lose_streak_diff(lose_streak_diff)
                .build()
            )
            
            player1_won : bool = player1_name == row["Winner"]
            player1.game_update(
                win = player1_won,
                rival = player2_name,
                match_date = row["Date"],
                comment = row["Comment"],
                court = row["Court"],
                tournament = row["Tournament"],
                surface = row["Surface"],
                round = row["Round"],
                series = row["Series"],
                sets_won = row["player1_sets"],
                games_played = games_played,
                rank = row["player1_Rank"],
                best_of = row["Best of"],
                games_won = player1_games_won,
                sets_played = row["player1_sets"] + row["player2_sets"],
            )
            player2.game_update(
                win = not player1_won,
                rival = player1_name,
                match_date = row["Date"],
                comment = row["Comment"],
                court = row["Court"],
                tournament = row["Tournament"],
                surface = row["Surface"],
                round = row["Round"],
                series = row["Series"],
                sets_won = row["player2_sets"],
                games_played = games_played,
                rank = row["player2_Rank"],
                best_of = row["Best of"],
                games_won = player2_games_won,
                sets_played = row["player1_sets"] + row["player2_sets"],
            )

        return matches