from dataclasses import dataclass, field
import enum

import numpy as np
from collections import defaultdict, deque
from datetime import datetime    

class WeightedRankingMethod(enum.Enum):
    INV = "inv"
    LOG = "log"
    INVSQRT = "invsqrt"

class WeightedPointsMethod(enum.Enum):
    EXP = "exp"
    SQUARE = "square"
    CUBE = "cube"

@dataclass
class PerformanceStats:
    matches : int = 0
    wins : int = 0

@dataclass
class PerformanceStatsWithRivals:
    stats : PerformanceStats = field(default_factory=PerformanceStats)
    rivals : defaultdict[str, PerformanceStats] = field(default_factory=lambda: defaultdict(PerformanceStats))


@dataclass
class PerformanceTracker:
    best_of : defaultdict[int, PerformanceStats] = field(default_factory=lambda: defaultdict(lambda : PerformanceStats()))
    courts : defaultdict[str, PerformanceStatsWithRivals] = field(default_factory=lambda: defaultdict(lambda : PerformanceStatsWithRivals()))
    surfaces : defaultdict[str, PerformanceStatsWithRivals] = field(default_factory=lambda: defaultdict(lambda : PerformanceStatsWithRivals()))

@dataclass
class MatchResult:
    date : datetime
    win : bool
    rank : int
    games_won : int
    games_played : int
    sets_won : int
    sets_played : int

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
        self.last_k_matches : deque[MatchResult] = deque(maxlen=k)
        self.win_streak : int = 0 
        self.lose_streak : int = 0
        self.performance = PerformanceTracker()

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

        self.performance.best_of[best_of].matches += 1
        self.last_match = {
            "date" : match_date,
            "games_played" : games_played
        }
        self.num_games += 1

        if win:
            self.performance.best_of[best_of].wins += 1
            self.wins[rival]["matches"] += 1
            self.wins[rival]["sets"] += sets_won
            self.win_streak += 1
            self.lose_streak = 0
        else:
            self.win_streak = 0
            self.lose_streak += 1

        self.h2h_matches[rival].append((match_date, win))
        self.last_k_matches.append(
            MatchResult(
                date=match_date,
                win=win,
                rank=rank,
                games_won=games_won,
                games_played=games_played,
                sets_won=sets_won,
                sets_played=sets_played
            )
        )

        self.performance.courts[court].stats.matches += 1
        self.performance.surfaces[surface].stats.matches += 1
        self.performance.courts[court].rivals[rival].matches += 1
        self.performance.surfaces[surface].rivals[rival].matches += 1
        if win:
            self.performance.courts[court].rivals[rival].wins += 1
            self.performance.surfaces[surface].rivals[rival].wins += 1
            self.performance.courts[court].rivals[rival].wins += 1
            self.performance.surfaces[surface].rivals[rival].wins += 1


    def get_weighted_ranking(self, current_ranking : int, method : WeightedRankingMethod) -> float:
        if method == WeightedRankingMethod.INV:
            return 1 / current_ranking
        elif method == WeightedRankingMethod.LOG:
            return -np.log(current_ranking)
        elif method == WeightedRankingMethod.INVSQRT:
            return 1 / np.sqrt(current_ranking)
        else:
            return np.exp(-current_ranking)


    def get_weighted_points(self, current_points : float, method : WeightedPointsMethod) -> float:
        if method == WeightedPointsMethod.EXP:
            return np.exp(current_points / 1000)  # Scale down points to avoid overflow
        elif method == WeightedPointsMethod.SQUARE:
            return current_points ** 2
        elif method == WeightedPointsMethod.CUBE:
            return current_points ** 3
        return current_points


    def get_win_rate(self) -> float :
        return len(self.wins) / self.num_games if self.num_games > 0 else 0.5


    def get_win_rate_by_best_of(self, best_of : int) -> float:
        """Return the player's win rate for best-of-3 or best-of-5 matches."""
        if best_of not in (3, 5):
            raise ValueError("best_of must be either 3 or 5")

        performance = self.performance.best_of[best_of]
        if performance.matches == 0:
            return 0.5
        
        return performance.wins / performance.matches


    def get_matches_played_last_days(
            self,
            reference_date : datetime,
            days : int = 21,
    ) -> int:
        """Return the number of matches played in the preceding time window."""
        if days < 0:
            raise ValueError("days must be non-negative")

        return sum(
            0 <= (reference_date - match_date.date).days <= days
            for match_date in self.last_k_matches
        )


    def get_last_k_games_win_rate(self) -> float:
        """Return the games won divided by games played in the last k matches."""
        games_won = sum(match.games_won for match in self.last_k_matches)
        games_lost = sum(match.games_played - match.games_won for match in self.last_k_matches)
        total_games = games_won + games_lost
        return games_won / total_games if total_games > 0 else 0.5


    def get_last_k_sets_win_rate(self) -> float:
        """Return the sets won divided by sets played in the last k matches."""
        sets_won = sum(match.sets_won for match in self.last_k_matches)
        sets_lost = sum(match.sets_played - match.sets_won for match in self.last_k_matches)
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
            rate = sum(match.win for match in self.last_k_matches) / n
            confidence = n / self.k 
            return rate * confidence + 0.5 * (1 - confidence)
        return sum(match.win for match in self.last_k_matches) / len(self.last_k_matches)


    def get_last_k_matches_rank_variation(self, current_rank: int) -> int:
        if not self.last_k_matches:
            return 0
        return current_rank - self.last_k_matches[0].rank

    
    def get_court_performance(
            self, 
            court : str
        ) -> float:
        "Return the win rate of the player on a specific court if any games are played, otherwise 0.5"

        court_performance : PerformanceStatsWithRivals = self.performance.courts[court]
        court_win_rate = court_performance.stats.wins / court_performance.stats.matches if court_performance.stats.matches > 0 else 0.5
        return court_win_rate


    def get_h2h_court_performance(
            self,
            court : str,
            rival : str,
    ) -> float:
        """Return this player's win rate against a rival on a specific court."""
        rival_performance : PerformanceStats = self.performance.courts[court].rivals[rival]
        return (
            rival_performance.wins / rival_performance.matches
            if rival_performance.matches > 0 else 0.5
        )


    def get_surface_performance(
            self,
            surface : str,
    ) -> float:
        
        "Return the win rate of the player on a specific surface if any games are played, otherwise 0.5"
        surface_performance : PerformanceStatsWithRivals = self.performance.surfaces[surface]
        surface_win_rate = surface_performance.stats.wins / surface_performance.stats.matches if surface_performance.stats.matches > 0 else 0.5
        return surface_win_rate


    def get_h2h_surface_performance(
            self,
            surface : str,
            rival : str,
    ) -> float:
        """Return this player's win rate against a rival on a specific surface."""
        rival_performance = self.performance.surfaces[surface].rivals[rival]
        return (
            rival_performance.wins / rival_performance.matches
            if rival_performance.matches > 0 else 0.5
        )
