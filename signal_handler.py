class SignalHandler:
    """A class to handle shutdown signals gracefully."""

    def __init__(self):
        self.shutdown_requested = False

    def __call__(self, signum, frame):
        print(f"\n[Main] Shutdown signal {signum} received. Finishing current tasks and saving...")
        self.shutdown_requested = True
