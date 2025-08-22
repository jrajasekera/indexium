import logging

logger = logging.getLogger(__name__)


class SignalHandler:
    """A class to handle shutdown signals gracefully."""

    def __init__(self):
        self.shutdown_requested = False

    def __call__(self, signum, frame):
        logger.info(
            "[Main] Shutdown signal %s received. Finishing current tasks and saving...",
            signum,
        )
        self.shutdown_requested = True
