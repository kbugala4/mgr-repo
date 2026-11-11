"""Advisory fit-budget accounting for the three local neural wrappers."""

import time


class BudgetGuard:
    def __init__(self, time_limit, name):
        self.started = time.monotonic()
        self.deadline = float("inf") if time_limit is None else self.started + time_limit
        self.time_limit = time_limit
        self.name = name
        self.completed_epochs = 0
        self.truncated = False

    def start_epoch(self):
        return time.monotonic()

    def check_batch(self):
        if time.monotonic() < self.deadline:
            return False
        if not self.completed_epochs:
            raise TimeoutError(f"{self.name} exceeded its fit budget before completing an epoch")
        self.truncated = True
        return True

    def finish_epoch(self, started, max_epochs, *, early_stopped=False):
        self.completed_epochs += 1
        now = time.monotonic()
        if (
            not early_stopped
            and self.completed_epochs < max_epochs
            and now + (now - started) > self.deadline
        ):
            self.truncated = True
        return self.truncated

    def result(self, batch_size):
        return {
            "budget_truncated": self.truncated,
            "completed_epochs": self.completed_epochs,
            "granted_seconds": self.time_limit,
            "fit_seconds_wrapper": time.monotonic() - self.started,
            "effective_batch_size": batch_size,
        }
