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
    rival_wins : defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class PerformanceTracker:
    best_of : defaultdict[int, PerformanceStats] = field(default_factory=lambda: defaultdict(lambda : PerformanceStats()))
    courts : defaultdict[str, PerformanceStatsWithRivals] = field(default_factory=lambda: defaultdict(lambda : PerformanceStatsWithRivals()))
    tournaments : defaultdict[str, PerformanceStatsWithRivals] = field(default_factory=lambda : defaultdict(lambda : PerformanceStatsWithRivals()))
    surfaces : defaultdict[str, PerformanceStatsWithRivals] = field(default_factory=lambda : defaultdict(lambda : PerformanceStatsWithRivals()))
    rounds : defaultdict[float, PerformanceStatsWithRivals] = field(default_factory=lambda: defaultdict(lambda : PerformanceStatsWithRivals()))
    series : defaultdict[float, PerformanceStatsWithRivals] = field(default_factory=lambda: defaultdict(lambda : PerformanceStatsWithRivals()))

@dataclass
class MatchResult:
    date : datetime
    non_completed : bool
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
            comment : str,
            court : str,
            tournament : str,
            surface : str,
            round : float,
            series : float,
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
                non_completed=comment.strip().lower() != "completed",
                win=win,
                rank=rank,
                games_won=games_won,
                games_played=games_played,
                sets_won=sets_won,
                sets_played=sets_played
            )
        )

        self.performance.courts[court].stats.matches += 1
        self.performance.tournaments[tournament].stats.matches += 1
        self.performance.surfaces[surface].stats.matches += 1
        self.performance.rounds[round].stats.matches += 1
        self.performance.series[series].stats.matches += 1
        if win:
            self.performance.courts[court].stats.wins += 1
            self.performance.tournaments[tournament].stats.wins += 1
            self.performance.surfaces[surface].stats.wins += 1
            self.performance.rounds[round].stats.wins += 1
            self.performance.series[series].stats.wins += 1
            self.performance.courts[court].rival_wins[rival] += 1
            self.performance.tournaments[tournament].rival_wins[rival] += 1
            self.performance.surfaces[surface].rival_wins[rival] += 1
            self.performance.rounds[round].rival_wins[rival] += 1
            self.performance.series[series].rival_wins[rival] += 1


    def get_last_k_non_completed_matches(self) -> float:
        """Return a recency-weighted non-completion score over the last k matches."""

        score = 0.0
        for position, match in enumerate(self.last_k_matches):
            if match.non_completed:
                score += 1 / (self.k - position + 1)
        return score


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
        return weighted_wins

    
    def get_fatigue(
            self, 
            match_date : datetime, 
            long_stop_days : int = 21, 
            max_games : int = 39
        ) -> float:
        """
        Return a recent workload and layoff-risk score in the range [0, 1].

        The score combines four components:

        1. Recent workload: every previous match contributes its games played,
           plus an additional load for each set above two. Its contribution
           decays exponentially over three recovery days, so recent matches
           matter more than older matches.
        2. Congestion: two or more matches within the previous three days
           increase the score.
        3. Short rest: each match played with fewer than two full recovery
           days contributes an additional penalty.
        4. Layoff risk: a stop longer than ``long_stop_days`` contributes a
           bounded risk term. This represents possible injury or loss of
           match readiness; it is not treated as physical workload.

        Non-completed matches receive a 10% workload increase because a
        retirement may indicate an underlying physical problem. The final
        score is a weighted sum:

            0.55 * recent workload
          + 0.20 * match congestion
          + 0.15 * short-rest penalty
          + 0.10 * long-stop risk

        ``max_games`` normalizes games played for best-of-three or
        best-of-five matches.
        """

        if not self.last_k_matches:
            return 0

        recovery_days = 3.0
        recent_load = 0.0
        short_rest = 0.0
        matches_within_three_days = 0

        for match in self.last_k_matches:
            days_since_match = (match_date - match.date).days
            workload = match.games_played / max_games + 0.1 * max(0, match.sets_played - 2)
            if match.non_completed:
                workload *= 1.1

            recent_load += workload * np.exp(-days_since_match / recovery_days)
            short_rest += max(0.0, (2.0 - days_since_match) / 2.0)
            matches_within_three_days += days_since_match <= 3

        normalized_load = min(1.0, recent_load)
        congestion = min(1.0, max(0, matches_within_three_days - 1) / 2.0)
        short_rest = min(1.0, short_rest)

        days_since_last_match = (match_date - self.last_k_matches[-1].date).days
        layoff_risk = (
            min(1.0, (days_since_last_match - long_stop_days) / long_stop_days)
            if days_since_last_match > long_stop_days else 0.0
        )

        return (
            0.55 * normalized_load
            + 0.20 * congestion
            + 0.15 * short_rest
            + 0.10 * layoff_risk
        )

    
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

    
    def get_court_win_rate(
            self, 
            court : str
        ) -> float:
        "Return the win rate of the player on a specific court if any games are played, otherwise 0.5"

        court_performance : PerformanceStatsWithRivals = self.performance.courts[court]
        court_win_rate = court_performance.stats.wins / court_performance.stats.matches if court_performance.stats.matches > 0 else 0.5
        return court_win_rate


    def get_tournament_win_rate(
            self,
            tournament : str,
    ) -> float:
        "Return the win rate of the player in a specific tournament if any games are played, otherwise 0.5"

        tournament_performance : PerformanceStatsWithRivals = self.performance.tournaments[tournament]
        tournament_win_rate = tournament_performance.stats.wins / tournament_performance.stats.matches if tournament_performance.stats.matches > 0 else 0.5
        return tournament_win_rate


    def get_h2h_tournament_wins(
            self,
            tournament : str,
            rival : str,
    ) -> float:
        """Return this player's win count against a rival in a specific tournament."""
        return self.performance.tournaments[tournament].rival_wins[rival]


    def get_h2h_court_wins(
            self,
            court : str,
            rival : str,
    ) -> float:
        """Return this player's win count against a rival on a specific court."""
        return self.performance.courts[court].rival_wins[rival]

    def get_surface_win_rate(
            self,
            surface : str,
    ) -> float:
        
        "Return the win rate of the player on a specific surface if any games are played, otherwise 0.5"
        surface_performance : PerformanceStatsWithRivals = self.performance.surfaces[surface]
        surface_win_rate = surface_performance.stats.wins / surface_performance.stats.matches if surface_performance.stats.matches > 0 else 0.5
        return surface_win_rate


    def get_h2h_surface_wins(
            self,
            surface : str,
            rival : str,
    ) -> float:
        """Return this player's win count against a rival on a specific surface."""
        return self.performance.surfaces[surface].rival_wins[rival] 


    def get_round_win_rate(self, round : float) -> float:
        """Return the win rate of the player in a specific round, otherwise 0.5."""
        round_performance = self.performance.rounds[round]
        return (
            round_performance.stats.wins / round_performance.stats.matches
            if round_performance.stats.matches > 0 else 0.5
        )


    def get_h2h_round_wins(self, round : float, rival : str) -> float:
        """Return this player's win count against a rival in a specific round."""
        return self.performance.rounds[round].rival_wins[rival]


    def get_series_win_rate(self, series : float) -> float:
        """Return the win rate of the player in a specific series, otherwise 0.5."""
        series_performance = self.performance.series[series]
        return (
            series_performance.stats.wins / series_performance.stats.matches
            if series_performance.stats.matches > 0 else 0.5
        )


    def get_h2h_series_wins(self, series : float, rival : str) -> float:
        """Return this player's win count against a rival in a specific series."""
        return self.performance.series[series].rival_wins[rival]
