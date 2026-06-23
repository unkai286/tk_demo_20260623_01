import time
from logging import INFO, getLogger

logger = getLogger(__name__)
logger.setLevel(INFO)


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} secondes")
        return result

    return wrapper
