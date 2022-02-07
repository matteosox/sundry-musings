"""Module for time utilities"""

import time
from collections.abc import Generator
from contextlib import contextmanager

__all__ = ["stopwatch"]


@contextmanager
def stopwatch() -> Generator[None, None, None]:
    """
    Simple context manager that prints the time it takes
    to execute a block of code.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        output = []
        if hours:
            output.append(f"{hours:.0f} hr, ")
        if minutes:
            output.append(f"{minutes:.0f} min, ")
        output.append(f"{seconds:.3f} sec")
        print("".join(output))
