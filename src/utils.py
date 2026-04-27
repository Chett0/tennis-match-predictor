from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime

RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"

class Player:

    def __init__(self, k = 5):
        self.last_match = None
        self.wins : dict[str, dict[str, int]] = defaultdict(lambda : {
                "matches" : 0,
                "sets" : 0
        })
        self.num_games : int = 0
        self.num_wins : int = 0
        self.last_k_matches : deque[int] = deque(maxlen=k)
        self.court_performance : dict[str, dict[str, int]] = defaultdict(lambda : {
            "games" : 0, 
            "wins" : 0
        })
        self.surface_performance : dict[str, dict[str, int]] = defaultdict(lambda : {
            "games" : 0, 
            "wins" : 0
        })

    def game_update(
            self,
            win : bool,
            rival : str,
            match_date : datetime,
            court : str,
            surface : str,
            sets_won : int = 0,
            games_played : int = 0,
        ):
        self.last_match = {
            "date" : match_date,
            "games_played" : games_played
        }
        self.num_games += 1
        self.last_k_matches.append(1 if win else 0)

        if win:
            self.num_wins += 1
            self.wins[rival]["matches"] += 1
            self.wins[rival]["sets"] += sets_won
        
        self.court_performance[court]["games"] += 1
        self.surface_performance[surface]["games"] += 1
        if win:
            self.court_performance[court]["wins"] += 1
            self.surface_performance[surface]["wins"] += 1



    def get_win_rate(self) -> float :
        return self.num_wins / self.num_games if self.num_games > 0 else 0.5
    
    def get_fatigue(self, match_date : datetime, long_stop_days : int = 15, max_games : int = 39) -> float:

        if self.last_match is None:
            return 1
        
        days_since_last_match = (match_date - self.last_match["date"]).days
        last_match_games = self.last_match["games_played"]

        if days_since_last_match == 0:
            return 1

        if days_since_last_match > long_stop_days:
            return long_stop_days / days_since_last_match
        else:
            return (last_match_games / max_games) / days_since_last_match
        
    def get_last_k_matches_win_rate(self) -> float:
        if not self.last_k_matches:
            return 0.5
        return sum(self.last_k_matches) / len(self.last_k_matches)
    
    def get_court_surface_performance(
            self, 
            court : str,
            surface : str
        ) -> tuple[float, float]:
        court_performance : dict[str, int] = self.court_performance[court]
        surface_performance : dict[str, int] = self.surface_performance[surface]

        court_win_rate = court_performance["wins"] / court_performance["games"] if court_performance["games"] > 0 else 0.5
        surface_win_rate = surface_performance["wins"] / surface_performance["games"] if surface_performance["games"] > 0 else 0.5

        return court_win_rate, surface_win_rate


@dataclass
class Match:
    round : int
    tournament : int
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
    
    def add_tournament(self, tournament : int):
        self.tournament : int = tournament
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
    
    def add_court_win_rate_diff(self, court_win_rate_diff : float):
        self.court_win_rate_diff : float = court_win_rate_diff
        return self
    
    def add_surface_win_rate_diff(self, surface_win_rate_diff : float):
        self.surface_win_rate_diff : float = surface_win_rate_diff
        return self

    def build(self):
        return Match(
            round = self.round,
            tournament = self.tournament,
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
            winner = self.winner
        )