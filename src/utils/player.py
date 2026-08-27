import enum

import numpy as np
from collections import defaultdict, deque
from datetime import datetime    


class WeightedRankingMethod(enum.Enum):
    INV = "inv"
    LOG = "log"
    INVSQRT = "invsqrt"


class Player:

    def __init__(self, k = 20):
        self.k = k
        self.last_match = None
        self.wins = defaultdict(lambda : {
                "matches" : 0,
                "sets" : 0,
        })
        self.h2h_matches : defaultdict[str, list[tuple[datetime, bool]]] = defaultdict(list)
        self.num_games : int = 0
        self.num_wins : int = 0
        self.match_results : list[tuple[datetime, bool]] = []
        self.best_of_performance : dict[int, dict[str, int]] = defaultdict(lambda : {
            "games" : 0,
            "wins" : 0,
        })
        self.last_k_matches : deque[int] = deque(maxlen=k)
        self.last_k_matches_rank: deque[int] = deque(maxlen=k)
        self.last_k_games : deque[tuple[int, int]] = deque(maxlen=k)
        self.last_k_sets : deque[tuple[int, int]] = deque(maxlen=k)
        self.win_streak : int = 0 
        self.lose_streak : int = 0
        self.court_performance : dict= defaultdict(lambda : {
            "games" : 0, 
            "wins" : 0,
            "rivals" : defaultdict(lambda : {
                "games" : 0,
                "wins" : 0
            })
        })
        self.surface_performance : dict = defaultdict(lambda : {
            "games" : 0, 
            "wins" : 0,
            "rivals" : defaultdict(lambda : {
                "games" : 0,
                "wins" : 0
            })
        })

    def game_update(
            self,
            win : bool,
            rival : str,
            match_date : datetime,
            court : str,
            surface : str,
            rank: int,
            sets_won : int,
            sets_played : int,
            games_won : int,
            games_played : int,
            best_of : int,
        ):
        """Update player stats with game information"""
        if best_of not in (3, 5):
            raise ValueError("best_of must be either 3 or 5")

        self.best_of_performance[best_of]["games"] += 1
        self.last_match = {
            "date" : match_date,
            "games_played" : games_played
        }
        self.num_games += 1
        self.match_results.append((match_date, win))

        if win:
            self.num_wins += 1
            self.best_of_performance[best_of]["wins"] += 1
            self.wins[rival]["matches"] += 1
            self.wins[rival]["sets"] += sets_won
            self.win_streak += 1
            self.lose_streak = 0
            self.last_k_matches.append(1)
        else:
            self.win_streak = 0
            self.lose_streak += 1
            self.last_k_matches.append(0)

        self.h2h_matches[rival].append((match_date, win))
        self.last_k_matches_rank.append(rank)
        self.last_k_games.append((games_won, games_played - games_won))
        self.last_k_sets.append((sets_won, sets_played - sets_won))

        self.court_performance[court]["games"] += 1
        self.surface_performance[surface]["games"] += 1
        self.court_performance[court]["rivals"][rival]["games"] += 1
        self.surface_performance[surface]["rivals"][rival]["games"] += 1
        if win:
            self.court_performance[court]["wins"] += 1
            self.surface_performance[surface]["wins"] += 1
            self.court_performance[court]["rivals"][rival]["wins"] += 1
            self.surface_performance[surface]["rivals"][rival]["wins"] += 1


    def get_weighted_ranking(self, current_ranking : int, method : WeightedRankingMethod) -> float:
        if method == WeightedRankingMethod.INV:
            return 1 / current_ranking
        elif method == WeightedRankingMethod.LOG:
            return -np.log(current_ranking)
        elif method == WeightedRankingMethod.INVSQRT:
            return 1 / np.sqrt(current_ranking)
        else:
            return np.exp(-current_ranking)


    def get_win_rate(self) -> float :
        return self.num_wins / self.num_games if self.num_games > 0 else 0.5


    def get_weighted_win_rate(
            self,
            reference_date : datetime,
            decay_days : float = 365,
    ) -> float:
        """Return the win rate with exponentially decaying recency weights."""
        if decay_days <= 0:
            raise ValueError("decay_days must be positive")

        weighted_wins = 0.0
        total_weight = 0.0
        for match_date, win in self.match_results:
            age_days = max(0, (reference_date - match_date).days)
            weight = np.exp(-age_days / decay_days)
            total_weight += weight
            if win:
                weighted_wins += weight

        return weighted_wins / total_weight if total_weight > 0 else 0.5


    def get_win_rate_by_best_of(self, best_of : int) -> float:
        """Return the player's win rate for best-of-3 or best-of-5 matches."""
        if best_of not in (3, 5):
            raise ValueError("best_of must be either 3 or 5")

        performance = self.best_of_performance[best_of]
        if performance["games"] == 0:
            return 0.5
        
        return performance["wins"] / performance["games"]


    def get_matches_played_last_days(
            self,
            reference_date : datetime,
            days : int = 21,
    ) -> int:
        """Return the number of matches played in the preceding time window."""
        if days < 0:
            raise ValueError("days must be non-negative")

        return sum(
            0 <= (reference_date - match_date[0]).days <= days
            for match_date in self.match_results
        )


    def get_last_k_games_win_rate(self) -> float:
        """Return the games won divided by games played in the last k matches."""
        games_won = sum(match[0] for match in self.last_k_games)
        games_lost = sum(match[1] for match in self.last_k_games)
        total_games = games_won + games_lost
        return games_won / total_games if total_games > 0 else 0.5


    def get_last_k_sets_win_rate(self) -> float:
        """Return the sets won divided by sets played in the last k matches."""
        sets_won = sum(match[0] for match in self.last_k_sets)
        sets_lost = sum(match[1] for match in self.last_k_sets)
        total_sets = sets_won + sets_lost
        return sets_won / total_sets if total_sets > 0 else 0.5


    def get_h2h_weighted_wins(
            self,
            rival : str,
            reference_date : datetime,
            decay_days : float = 365,
    ) -> float:
        """Return wins against a rival with exponentially decaying recency weights."""
        if decay_days <= 0:
            raise ValueError("decay_days must be positive")

        weighted_wins = 0.0
        for match_date, win in self.h2h_matches[rival]:
            if win:
                age_days = max(0, (reference_date - match_date).days)
                weighted_wins += np.exp(-age_days / decay_days)
        return float(weighted_wins)

    
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
        elif len(self.last_k_matches) < self.k:
            n = len(self.last_k_matches)
            rate = sum(self.last_k_matches) / n
            confidence = n / self.k 
            return rate * confidence + 0.5 * (1 - confidence)
        return sum(self.last_k_matches) / len(self.last_k_matches)


    def get_last_k_matches_rank_variation(self, current_rank: int) -> int:
        if not self.last_k_matches_rank:
            return 0
        return current_rank - self.last_k_matches_rank[0]

    
    def get_court_performance(
            self, 
            court : str
        ) -> float:
        "Return the win rate of the player on a specific court if any games are played, otherwise 0.5"

        court_performance : dict[str, int] = self.court_performance[court]
        court_win_rate = court_performance["wins"] / court_performance["games"] if court_performance["games"] > 0 else 0.5
        return court_win_rate


    def get_h2h_court_performance(
            self,
            court : str,
            rival : str,
    ) -> float:
        """Return this player's win rate against a rival on a specific court."""
        rival_performance = self.court_performance[court]["rivals"][rival]
        return (
            rival_performance["wins"] / rival_performance["games"]
            if rival_performance["games"] > 0 else 0.5
        )


    def get_surface_performance(
            self,
            surface : str,
    ) -> float:
        
        "Return the win rate of the player on a specific surface if any games are played, otherwise 0.5"
        surface_performance : dict[str, int] = self.surface_performance[surface]
        surface_win_rate = surface_performance["wins"] / surface_performance["games"] if surface_performance["games"] > 0 else 0.5
        return surface_win_rate


    def get_h2h_surface_performance(
            self,
            surface : str,
            rival : str,
    ) -> float:
        """Return this player's win rate against a rival on a specific surface."""
        rival_performance = self.surface_performance[surface]["rivals"][rival]
        return (
            rival_performance["wins"] / rival_performance["games"]
            if rival_performance["games"] > 0 else 0.5
        )
