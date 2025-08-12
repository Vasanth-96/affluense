from utils.async_logger import logger
import concurrent.futures

THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="sentiment_analysis"
)

def shutdown_thread_pool():
    """Cleanup function to shutdown the thread pool"""
    THREAD_POOL.shutdown(wait=True)
    logger.info("Thread pool shut down")
