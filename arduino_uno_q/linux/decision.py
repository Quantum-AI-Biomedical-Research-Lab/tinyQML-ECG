from collections import deque

class DecisionSmoother:
    
    def __init__(self, k: int = 2, n: int = 3):
        self.k, self.n = k, n
        self.history = deque(maxlen=n)

    def update(self, abnormal):
        if abnormal is None:
            self.history.clear()
            return "POOR_SIGNAL"
        self.history.append(bool(abnormal))
        if len(self.history) < self.n:
            return "WARMUP"
        return "ABNORMAL" if sum(self.history) >= self.k else "NORMAL"