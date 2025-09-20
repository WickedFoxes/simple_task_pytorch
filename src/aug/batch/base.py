from src.registry import register

class BatchAugBase:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, y):
        # 반환은 (x’, (y_a, y_b, lam), used)
        raise NotImplementedError
