from tqdm import tqdm
import random, time
from src.logger.base import LoggerBase
from src.registry import register

@register("logger", "simple")
class SimpleLogger(LoggerBase):
    def __init__(self):
        self.metrics_history = []
        self.params = {}

    def log_metrics(self, metrics: dict, step: int = None):
        """메트릭을 로그에 출력"""
        if step is not None:
            print(f"[Metrics][Step {step}] {metrics}")
        else:
            print(f"[Metrics] {metrics}")
        self.metrics_history.append((step, metrics))

    def log_params(self, params: dict):
        """하이퍼파라미터를 로그에 출력"""
        print(f"[Params] {params}")
        self.params.update(params)

    def finalize(self, status="success"):
        """로그 세션 종료"""
        print(f"[Finalize] Training finished with status: {status}")
        print(f"Logged Params: {self.params}")
        print(f"Total Metrics Logged: {len(self.metrics_history)}")