import csv
import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class EpisodeRecord:
    episode: int
    total_return: float
    length: int
    success: bool
    epsilon: float = 0.0
    loss: float = 0.0


@dataclass
class EvalRecord:
    episode: int
    avg_return: float
    win_rate: float
    avg_length: float
    avg_success_length: float
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

    def __init__(self, log_dir: str, agent_name: str = "agent"):
        self.log_dir = log_dir
        self.agent_name = agent_name
        self.episodes: list[EpisodeRecord] = []
        self.evals: list[EvalRecord] = []
        self.baselines: list[BaselineRecord] = []
        os.makedirs(log_dir, exist_ok=True)

    def log_episode(
        self,
        episode: int,
        total_return: float,
        length: int,
        success: bool,
        epsilon: float = 0.0,
        loss: float = 0.0,
    ) -> None:
        self.episodes.append(
            EpisodeRecord(
                episode=episode,
                total_return=total_return,
                length=length,
                success=success,
                epsilon=epsilon,
                loss=loss,
            )
        )

    def log_eval(self, record: EvalRecord) -> None:
        self.evals.append(record)

    def log_baseline(
        self,
        name: str,
        result: dict,
        n_episodes: int,
        max_steps: int,
    ) -> None:
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
        path = path or os.path.join(self.log_dir, f"{self.agent_name}_training.csv")
        if not self.episodes:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.episodes[0]).keys()))
            writer.writeheader()
            for ep in self.episodes:
                writer.writerow(asdict(ep))

    def save_eval_json(self, path: str | None = None) -> None:
        path = path or os.path.join(self.log_dir, f"{self.agent_name}_eval.json")
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.evals], f, indent=2)

    def save_baseline_json(self, path: str | None = None) -> None:
        path = path or os.path.join(self.log_dir, f"{self.agent_name}_baselines.json")
        with open(path, "w") as f:
            json.dump([asdict(b) for b in self.baselines], f, indent=2)

    def get_recent_stats(self, window: int = 100) -> dict:
        recent = self.episodes[-window:]
        if not recent:
            return {}
        returns = [e.total_return for e in recent]
        lengths = [e.length for e in recent]
        successes = [e.success for e in recent]
        loss = [e.loss for e in recent]
        return {
            "avg_return": sum(returns) / len(returns),
            "avg_length": sum(lengths) / len(lengths),
            "win_rate": sum(successes) / len(successes),
            "avg_loss": sum(loss) / len(loss),
        }
