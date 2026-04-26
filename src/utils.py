from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

class Player:

    def __init__(self):
        self.last_match = None
        self.wins : dict[str, int] = defaultdict(int)
        self.num_games : int = 0
        self.num_wins : int = 0

    def game_update(
            self,
            win : bool,
            rival : str,
            match_date : datetime
        ):
        self.last_match = match_date
        self.num_games += 1

        if win:
            self.num_wins += 1
            self.wins[rival] += 1 


    def get_win_rate(self) -> float :
        return self.num_wins / self.num_games if self.num_games > 0 else 0.5
    
    def get_fatigue(self, match_date : datetime) -> float:
        if self.last_match is None:
            return 0
        days_since_last_match = (match_date - self.last_match).days

        if days_since_last_match > 15:
            return 1 - (15 / days_since_last_match)
        else:
            return 1 / days_since_last_match
    


@dataclass
class Match:
    player_0 : str
    player_1 : str
    rank_diff : int
    h2h_diff : int
    win_rate_diff : float
    max_bet_diff : float
    avg_bet_diff : float
    fatigue_diff : float
    winner : int


class MatchBuilder:

    def add_first_player(self, player_name : str):
        self.player_0 : str = player_name
        return self
    
    def add_second_player(self, player_name : str):
        self.player_1 : str = player_name
        return self
    
    def add_rank_diff(self, rank_diff : int):
        self.rank_diff : int = rank_diff
        return self
    
    def add_h2h_diff(self, h2h_diff : int):
        self.h2h_diff : int = h2h_diff
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

    def build(self):
        return Match(
            player_0 = self.player_0,
            player_1 = self.player_1,
            rank_diff = self.rank_diff,
            h2h_diff = self.h2h_diff,
            win_rate_diff = self.win_rate_diff,
            max_bet_diff = self.max_bet_diff,
            avg_bet_diff = self.avg_bet_diff,
            fatigue_diff = self.fatigue_diff,
            winner = self.winner
        )