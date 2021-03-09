import sys
import time


class EventTimer:
    def __init__(self, name="", verbose=True):
        self._name = name
        self._verbose = verbose

    def __enter__(self):
        if self._verbose:
            print("-" * 12, f"Start `{self._name[:30]}`", "-" * 12, file=sys.stderr)

        self._timestamp = time.time()
        return self

    def __exit__(self, type, value, traceback):
        elapsed_time = time.time() - self._timestamp
        if self._verbose:
            print(
                "-" * 12,
                f"End `{self._name[:30]}` ({elapsed_time:.3f} s)",
                "-" * 12,
                file=sys.stderr,
            )

    def get_time(self):
        return time.time() - self._timestamp


def timer(func):
    def wrapper(*args, **kwargs):
        with EventTimer(verbose=False) as et:
            ret = func(*args, **kwargs)
            return (et.get_time(), ret)

    return wrapper
