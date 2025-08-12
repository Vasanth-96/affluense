import logging
import sys
from aiologger import Logger
from aiologger.formatters.base import Formatter
from aiologger.handlers.streams import AsyncStreamHandler

# Async logger for async operations
logger = Logger(
    name="async_operations",
    level='INFO',
)

async_handler = AsyncStreamHandler(stream=sys.stdout)
async_formatter = Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
async_handler.formatter = async_formatter
logger.add_handler(async_handler)

# Thread-safe logger for threaded operations
thread_logger = logging.getLogger('thread_operations')
thread_handler = logging.StreamHandler(sys.stdout)
thread_formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
thread_handler.setFormatter(thread_formatter)
thread_logger.addHandler(thread_handler)
thread_logger.setLevel(logging.INFO)

# Prevent double logging
thread_logger.propagate = False
