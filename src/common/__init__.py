import time


def now():
    return time.monotonic()


def dur(start, end=None):
    end = end if end is not None else now()
    return f"{end - start:.2f}s"


def stamp():
    return time.strftime('%H:%M:%S')
