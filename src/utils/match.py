from dataclasses import dataclass

@dataclass
class Match:
    round : int
    series : int
    rank_diff : int
    weighted_ranking_diff : float
    h2h_diff : int
    sets_h2h_diff : int
    win_rate_diff : float
    max_bet_diff : float
    avg_bet_diff : float
    fatigue_diff : float
    last_k_matches_win_rate_diff : float
    last_k_matches_rank_variation_diff : int
    win_streak_diff : int
    lose_streak_diff : int
    court_win_rate_diff : float
    surface_win_rate_diff : float
    h2h_weighted_diff : float = 0.0
    h2h_court_diff : float = 0.0
    h2h_surface_diff : float = 0.0
    win_rate_bestof3_diff : float = 0.0
    win_rate_bestof5_diff : float = 0.0
    number_recent_matches_diff : int = 0
    last_k_games_win_rate_diff : float = 0.0
    last_k_sets_win_rate_diff : float = 0.0


class MatchBuilder:

    def add_round(self, round : int):
        self.round : int = round
        return self
    
    def add_series(self, series : int):
        self.series : int = series
        return self

    def add_rank_diff(self, rank_diff : int):
        self.rank_diff : int = rank_diff
        return self

    def add_weighted_ranking_diff(self, weighted_ranking_diff : float):
        self.weighted_ranking_diff : float = weighted_ranking_diff
        return self
    
    def add_h2h_diff(self, h2h_diff : int):
        self.h2h_diff : int = h2h_diff
        return self
    
    def add_sets_h2h_diff(self, sets_h2h_diff : int):
        self.sets_h2h_diff : int = sets_h2h_diff
        return self

    def add_h2h_weighted_diff(self, h2h_weighted_diff : float):
        self.h2h_weighted_diff : float = h2h_weighted_diff
        return self

    def add_h2h_court_diff(self, h2h_court_diff : float):
        self.h2h_court_diff : float = h2h_court_diff
        return self

    def add_h2h_surface_diff(self, h2h_surface_diff : float):
        self.h2h_surface_diff : float = h2h_surface_diff
        return self

    def add_win_rate_bestof3_diff(self, win_rate_bestof3_diff : float):
        self.win_rate_bestof3_diff : float = win_rate_bestof3_diff
        return self

    def add_win_rate_bestof5_diff(self, win_rate_bestof5_diff : float):
        self.win_rate_bestof5_diff : float = win_rate_bestof5_diff
        return self

    def add_number_recent_matches_diff(self, number_recent_matches_diff : int):
        self.number_recent_matches_diff : int = number_recent_matches_diff
        return self

    def add_last_k_games_win_rate_diff(self, last_k_games_win_rate_diff : float):
        self.last_k_games_win_rate_diff : float = last_k_games_win_rate_diff
        return self

    def add_last_k_sets_win_rate_diff(self, last_k_sets_win_rate_diff : float):
        self.last_k_sets_win_rate_diff : float = last_k_sets_win_rate_diff
        return self
    
    def add_win_rate_diff(self, win_rate_diff : float):
        self.win_rate_diff : float = win_rate_diff
        return self
    
    def add_max_bet_diff(self, max_bet_diff : float):
        self.max_bet_diff : float = max_bet_diff
        return self
    
    def add_avg_bet_diff(self, avg_bet_diff : float):
        self.avg_bet_diff : float = avg_bet_diff
        return self
    
    def add_fatigue_diff(self, fatigue_diff : float):
        self.fatigue_diff : float = fatigue_diff
        return self
    
    def add_last_k_matches_win_rate_diff(self, last_k_matches_win_rate_diff : float):
        self.last_k_matches_win_rate_diff : float = last_k_matches_win_rate_diff
        return self

    def add_last_k_matches_rank_variation_diff(self, last_k_matches_rank_variation_diff : int):
        self.last_k_matches_rank_variation_diff : int = last_k_matches_rank_variation_diff
        return self
    
    def add_win_streak_diff(self, win_streak_diff : int):
        self.win_streak_diff : int = win_streak_diff
        return self
    
    def add_lose_streak_diff(self, lose_streak_diff : int):
        self.lose_streak_diff : int = lose_streak_diff
        return self
    
    def add_court_win_rate_diff(self, court_win_rate_diff : float):
        self.court_win_rate_diff : float = court_win_rate_diff
        return self
    
    def add_surface_win_rate_diff(self, surface_win_rate_diff : float):
        self.surface_win_rate_diff : float = surface_win_rate_diff
        return self

    def build(self):
        return Match(
            round = self.round,
            series = self.series,
            rank_diff = self.rank_diff,
            h2h_diff = self.h2h_diff,
            sets_h2h_diff = self.sets_h2h_diff,
            win_rate_diff = self.win_rate_diff,
            max_bet_diff = self.max_bet_diff,
            avg_bet_diff = self.avg_bet_diff,
            fatigue_diff = self.fatigue_diff,
            last_k_matches_win_rate_diff = self.last_k_matches_win_rate_diff,
            last_k_matches_rank_variation_diff = self.last_k_matches_rank_variation_diff,
            weighted_ranking_diff = self.weighted_ranking_diff,
            court_win_rate_diff = self.court_win_rate_diff,
            surface_win_rate_diff = self.surface_win_rate_diff,
            win_streak_diff = self.win_streak_diff,
            lose_streak_diff = self.lose_streak_diff,
            h2h_weighted_diff = self.h2h_weighted_diff,
            h2h_court_diff = self.h2h_court_diff,
            h2h_surface_diff = self.h2h_surface_diff,
            win_rate_bestof3_diff = self.win_rate_bestof3_diff,
            win_rate_bestof5_diff = self.win_rate_bestof5_diff,
            number_recent_matches_diff = self.number_recent_matches_diff,
            last_k_games_win_rate_diff = self.last_k_games_win_rate_diff,
            last_k_sets_win_rate_diff = self.last_k_sets_win_rate_diff,
        )