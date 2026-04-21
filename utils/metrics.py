import csv
import json
import os
from collections import deque
from dataclasses import dataclass, asdict


@dataclass
class EpisodeRecord:
    episode: int
    total_return: float
    length: int
    success: bool
    epsilon: float = 0.0
    loss: float = 0.0
    non_zero_reward_frac: float = 0.0
    terminal_frac: float = 0.0
    td_abs_zero: float = 0.0
    td_abs_pos: float = 0.0
    td_abs_neg: float = 0.0


@dataclass
class EvalRecord:
    episode: int
    avg_return: float
    sem_return: float
    win_rate: float
    sem_win_rate: float
    avg_length: float
    sem_length: float
    avg_success_length: float
    sem_success_length: float
    std_length: float


@dataclass
class BaselineRecord:
    name: str
    avg_return: float
    win_rate: float
    avg_length: float
    n_episodes: int
    max_steps: int


class MetricsLogger:
    """Tracks training and evaluation metrics, writes CSV and JSON outputs."""

    def __init__(self, log_dir: str, log_interval: int, agent_name: str = "agent"):
        """Initialize Metrics Logger.

        Initializes in-memory metric stores and ensures output directory exists.

        Args:
            log_dir (str): Directory for persisted metric files.
            log_interval (int): Number of recent episodes to keep in memory for
            stats
            agent_name (str): Prefix used for output file names.

        Returns:
            None: Logger state is initialized in place.
        """
        self.log_dir = log_dir
        self.agent_name = agent_name
        # Bounded deque: oldest records are dropped once the window is full,
        # preventing unbounded RAM growth during long training runs.
        self.episodes: deque[EpisodeRecord] = deque(maxlen=log_interval)
        self.evals: list[EvalRecord] = []
        self.baselines: list[BaselineRecord] = []
        self._csv_path: str = os.path.join(log_dir, f"{agent_name}_training.csv")
        self._csv_initialized: bool = False
        os.makedirs(log_dir, exist_ok=True)

    def log_episode(
        self,
        episode: int,
        total_return: float,
        length: int,
        success: bool,
        epsilon: float = 0.0,
        loss: float = 0.0,
        non_zero_reward_frac: float = 0.0,
        terminal_frac: float = 0.0,
        td_abs_zero: float = 0.0,
        td_abs_pos: float = 0.0,
        td_abs_neg: float = 0.0,
    ) -> None:
        """Log Training Episode.

        Appends a per-episode training metric record and flushes it to the CSV
        file immediately so that old records can be evicted from the in-memory
        deque without losing data.

        Args:
            episode (int): Episode index.
            total_return (float): Episode return.
            length (int): Episode length in steps.
            success (bool): Whether the episode solved the task.
            epsilon (float): Exploration value for the episode.
            loss (float): Average optimization loss.
            non_zero_reward_frac (float): Fraction of non-zero rewards in update
                batches.
            terminal_frac (float): Fraction of terminal transitions in update
                batches.
            td_abs_zero (float): Mean absolute TD error for zero-reward
                transitions.
            td_abs_pos (float): Mean absolute TD error for positive-reward
                transitions.
            td_abs_neg (float): Mean absolute TD error for negative-reward
                transitions.

        Returns:
            None: Record is appended to internal storage and written to disk.
        """
        record = EpisodeRecord(
            episode=episode,
            total_return=total_return,
            length=length,
            success=success,
            epsilon=epsilon,
            loss=loss,
            non_zero_reward_frac=non_zero_reward_frac,
            terminal_frac=terminal_frac,
            td_abs_zero=td_abs_zero,
            td_abs_pos=td_abs_pos,
            td_abs_neg=td_abs_neg,
        )
        self.episodes.append(record)
        self._append_episode_to_csv(record)

    def _append_episode_to_csv(self, record: EpisodeRecord) -> None:
        """Incrementally write one episode record to the CSV file.

        Opens the file in append mode so that old records do not need to be
        kept in memory. Writes the header on the first call.

        Args:
            record (EpisodeRecord): The record to persist.

        Returns:
            None: Record is written to disk.
        """
        mode = "w" if not self._csv_initialized else "a"
        with open(self._csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(record).keys()))
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(asdict(record))

    def log_eval(self, record: EvalRecord) -> None:
        """Log Evaluation Record.

        Appends one evaluation summary record.

        Args:
            record (EvalRecord): Evaluation metrics record.

        Returns:
            None: Record is appended to internal storage.
        """
        self.evals.append(record)

    def log_baseline(
        self,
        name: str,
        result: dict,
        n_episodes: int,
        max_steps: int,
    ) -> None:
        """Log Baseline Metrics.

        Stores baseline evaluation metrics for later persistence.

        Args:
            name (str): Baseline name.
            result (dict): Baseline metric dictionary.
            n_episodes (int): Number of baseline episodes.
            max_steps (int): Maximum steps per episode during baseline run.

        Returns:
            None: Baseline record is appended to internal storage.
        """
        self.baselines.append(
            BaselineRecord(
                name=name,
                avg_return=result["avg_return"],
                win_rate=result["win_rate"],
                avg_length=result["avg_length"],
                n_episodes=n_episodes,
                max_steps=max_steps,
            )
        )

    def save_csv(self, path: str | None = None) -> None:
        """Save Training CSV.

        No-op: episode records are now written incrementally to disk by
        ``log_episode`` so there is nothing left to flush here. The ``path``
        argument is accepted for backward compatibility but ignored when
        incremental writing is active.

        Args:
            path (str | None): Ignored (kept for backward compatibility).

        Returns:
            None
        """
        # Data is already on disk via _append_episode_to_csv; nothing to do.

    def save_eval_json(self, path: str | None = None) -> None:
        """Save Evaluation JSON.

        Persists evaluation records as JSON.

        Args:
            path (str | None): Optional output path. Default path is derived
                from logger settings.

        Returns:
            None: File is written to disk.
        """
        path = path or os.path.join(self.log_dir, f"{self.agent_name}_eval.json")
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.evals], f, indent=2)

    def get_recent_stats(self) -> dict:
        """Compute Recent Aggregate Stats.

        Computes moving-window averages over recent episode records.

        Args:
            window (int): Number of recent episodes to aggregate.

        Returns:
            dict: Aggregated metric dictionary for recent episodes.
        """
        recent = list(self.episodes)
        if not recent:
            return {}
        returns = [e.total_return for e in recent]
        lengths = [e.length for e in recent]
        successes = [e.success for e in recent]
        loss = [e.loss for e in recent]
        non_zero_reward_frac = [e.non_zero_reward_frac for e in recent]
        terminal_frac = [e.terminal_frac for e in recent]
        td_abs_zero = [e.td_abs_zero for e in recent]
        td_abs_pos = [e.td_abs_pos for e in recent]
        td_abs_neg = [e.td_abs_neg for e in recent]
        return {
            "avg_return": sum(returns) / len(returns),
            "avg_length": sum(lengths) / len(lengths),
            "win_rate": sum(successes) / len(successes),
            "avg_loss": sum(loss) / len(loss),
            "avg_non_zero_reward_frac": sum(non_zero_reward_frac)
            / len(non_zero_reward_frac),
            "avg_terminal_frac": sum(terminal_frac) / len(terminal_frac),
            "avg_td_abs_zero": sum(td_abs_zero) / len(td_abs_zero),
            "avg_td_abs_pos": sum(td_abs_pos) / len(td_abs_pos),
            "avg_td_abs_neg": sum(td_abs_neg) / len(td_abs_neg),
        }
