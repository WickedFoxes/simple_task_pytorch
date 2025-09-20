from tqdm import tqdm
import random, time
from src.logger.base import LoggerBase
from src.registry import register

@register("logger", "simple")
class SimpleLogger(LoggerBase):
    def __init__(self):
        self.best_acc = 0.0
        self.params = {}
        self.metrics_history = []

    def log_params(self, params: dict):
        """하이퍼파라미터 출력"""
        print(f"[Params] {params}")
        self.params.update(params)

    def log_metrics(self, metrics: dict, epoch: int, elapsed_time: float):
        """
        metrics: {
            "train_loss": float,
            "train_acc": float,
            "valid_loss": float,
            "valid_acc": float,
            "lr": float
        }
        """
        # 최고 정확도 갱신 시 출력
        if metrics["valid_acc"] > self.best_acc:
            self.best_acc = metrics["valid_acc"]
            print(f"New best accuracy: {self.best_acc:.4f} at epoch {epoch}")

        # 로그 라인 출력
        print(
            f"[Epoch{epoch:03d}] "
            f"train_loss={metrics['train_loss']:.4f} acc={metrics['train_acc']:.4f} | "
            f"valid_loss={metrics['valid_loss']:.4f} acc={metrics['valid_acc']:.4f} | "
            f"lr={metrics['lr']:.5f} | time={elapsed_time:.2f}s "
        )

        # 기록 저장
        self.metrics_history.append((epoch, metrics, elapsed_time))

    def finalize(self, status="success"):
        print(f"[Finalize] Training finished with status: {status}")
        print(f"Best Accuracy: {self.best_acc:.4f}")
        print(f"Logged Params: {self.params}")
        print(f"Total Epochs Logged: {len(self.metrics_history)}")