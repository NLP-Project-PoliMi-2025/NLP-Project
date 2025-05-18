import sys
import time
import contextlib
from itertools import cycle


@contextlib.contextmanager
def process_runindicator(desc="Process"):
    spinner = cycle(["|", "/", "-", "\\"])  # Spinner animation characters
    is_running = True

    def spinner_task():
        while is_running:
            sys.stdout.write(f"\r{desc}: {next(spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)  # Adjust the speed of the spinner

    try:
        # Start the spinner in a separate thread
        import threading
        spinner_thread = threading.Thread(target=spinner_task)
        spinner_thread.start()

        yield  # Yield control back to the wrapped process

    finally:
        # End spinner and show completion checkmark
        is_running = False
        spinner_thread.join()
        sys.stdout.write(f"\r{desc}: ✔\n")
        sys.stdout.flush()
