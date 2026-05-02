from dataclasses import dataclass
from datetime import datetime

@dataclass
class Match:
    round : int
    series : int
    date : datetime
    rank_diff : int
    h2h_diff : int
    sets_h2h_diff : int
    win_rate_diff : float
    max_bet_diff : float
    avg_bet_diff : float
    fatigue_diff : float
    last_k_matches_win_rate_diff : float
    win_streak_diff : int
    lose_streak_diff : int
    court_win_rate_diff : float
    surface_win_rate_diff : float
    winner : int

class MatchBuilder:

    def add_round(self, round : int):
        self.round : int = round
        return self
    
    def add_series(self, series : int):
        self.series : int = series
        return self
    
    def add_date(self, date : datetime):
        self.date : datetime = date
        return self

    def add_rank_diff(self, rank_diff : int):
        self.rank_diff : int = rank_diff
        return self
    
    def add_h2h_diff(self, h2h_diff : int):
        self.h2h_diff : int = h2h_diff
        return self
    
    def add_sets_h2h_diff(self, sets_h2h_diff : int):
        self.sets_h2h_diff : int = sets_h2h_diff
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
    
    def add_winner(self, winner : int):
        self.winner : int = winner
        return self
    
    def add_fatigue_diff(self, fatigue_diff : float):
        self.fatigue_diff : float = fatigue_diff
        return self
    
    def add_last_k_matches_win_rate_diff(self, last_k_matches_win_rate_diff : float):
        self.last_k_matches_win_rate_diff : float = last_k_matches_win_rate_diff
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
            date = self.date,
            rank_diff = self.rank_diff,
            h2h_diff = self.h2h_diff,
            sets_h2h_diff = self.sets_h2h_diff,
            win_rate_diff = self.win_rate_diff,
            max_bet_diff = self.max_bet_diff,
            avg_bet_diff = self.avg_bet_diff,
            fatigue_diff = self.fatigue_diff,
            last_k_matches_win_rate_diff = self.last_k_matches_win_rate_diff,
            court_win_rate_diff = self.court_win_rate_diff,
            surface_win_rate_diff = self.surface_win_rate_diff,
            win_streak_diff = self.win_streak_diff,
            lose_streak_diff = self.lose_streak_diff,
            winner = self.winner
        )