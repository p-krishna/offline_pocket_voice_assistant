import time


def now():
    return time.monotonic()


def dur(start, end=None):
    end = end if end is not None else now()
    return f"{end - start:.2f}s"


def stamp():
    return time.strftime('%H:%M:%S')

class _LatencyTracer:
    def __init__(self): self._m = {}
    def reset(self): self._m.clear()
    def mark(self, k): self._m[k] = time.perf_counter()
    def report(self):
        keys = list(self._m)
        return "  ".join(
            f"{keys[i-1]}→{keys[i]}={round((self._m[keys[i]]-self._m[keys[i-1]])*1000)}ms"
            for i in range(1, len(keys))
        )