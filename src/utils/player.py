from collections import defaultdict, deque
from datetime import datetime    

class Player:

    def __init__(self, k = 20):
        self.last_match = None
        self.wins = defaultdict(lambda : {
                "matches" : 0,
                "sets" : 0,
        })
        self.num_games : int = 0
        self.num_wins : int = 0
        self.last_k_matches : deque[int] = deque(maxlen=k)
        self.win_streak : int = 0 
        self.lose_streak : int = 0
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
        """Update player stats with game information"""

        self.last_match = {
            "date" : match_date,
            "games_played" : games_played
        }
        self.num_games += 1

        if win:
            self.num_wins += 1
            self.wins[rival]["matches"] += 1
            self.wins[rival]["sets"] += sets_won
            self.win_streak += 1
            self.lose_streak = 0
            self.last_k_matches.append(1)
        else:
            self.win_streak = 0
            self.lose_streak += 1
            self.last_k_matches.append(0)

        self.court_performance[court]["games"] += 1
        self.surface_performance[surface]["games"] += 1
        if win:
            self.court_performance[court]["wins"] += 1
            self.surface_performance[surface]["wins"] += 1


    def get_weighted_ranking(self, current_ranking : int) -> float:
        return current_ranking**(1.5)


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

    
    def get_court_performance(
            self, 
            court : str
        ) -> float:
        "Return the win rate of the player on a specific court if any games are played, otherwise 0.5"

        court_performance : dict[str, int] = self.court_performance[court]
        court_win_rate = court_performance["wins"] / court_performance["games"] if court_performance["games"] > 0 else 0.5
        return court_win_rate


    def get_surface_performance(
            self,
            surface : str,
    ) -> float:
        
        "Return the win rate of the player on a specific surface if any games are played, otherwise 0.5"
        surface_performance : dict[str, int] = self.surface_performance[surface]
        surface_win_rate = surface_performance["wins"] / surface_performance["games"] if surface_performance["games"] > 0 else 0.5
        return surface_win_rate


