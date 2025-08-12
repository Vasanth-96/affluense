import logging
import sys

# Thread-safe logger for synchronous and threaded operations
sync_logger = logging.getLogger('sync_operations')
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
sync_logger.addHandler(handler)
sync_logger.setLevel(logging.INFO)

# Prevent double logging if parent loggers exist
sync_logger.propagate = False
