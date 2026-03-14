"""Live training progress widget — polls run.log and renders stats."""

import os
import re

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

STEP_RE = re.compile(
    r"step\s+\d+\s+\(([\d.]+)%\)\s+\|\s+loss:\s+([\d.]+).*?"
    r"tok/sec:\s+([\d,]+).*?remaining:\s+(\d+)s"
)


class TrainingProgress(Widget):
    """Polls run.log and renders a progress bar with loss/tps/remaining."""

    DEFAULT_CSS = """
    TrainingProgress {
        height: 1;
        margin: 0 0 0 1;
    }
    """

    pct: reactive[float] = reactive(0.0)
    loss: reactive[str] = reactive("")
    tps: reactive[str] = reactive("")
    remaining: reactive[str] = reactive("")

    def __init__(self, log_path: str) -> None:
        super().__init__()
        self._log_path = log_path

    def on_mount(self) -> None:
        self._poll_timer = self.set_interval(1.0, self._poll)

    def render(self) -> Text:
        if self.pct == 0:
            return Text("▶ Training...", style="bold green")
        bar_width = 20
        filled = int(bar_width * self.pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        return Text.assemble(
            (bar, "green"),
            (f" {self.pct:.0f}%", "bold"),
            (f"  loss {self.loss}  {self.tps} tok/s  {self.remaining}s left", "dim"),
        )

    def _poll(self) -> None:
        if not os.path.exists(self._log_path):
            return
        try:
            with open(self._log_path, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 4096))
                tail = f.read().decode("utf-8", errors="replace")
            for seg in reversed(tail.replace("\r", "\n").split("\n")):
                m = STEP_RE.search(seg.strip())
                if m:
                    self.pct = float(m.group(1))
                    self.loss = m.group(2)
                    self.tps = m.group(3)
                    self.remaining = m.group(4)
                    break
        except Exception:
            pass

    def stop(self) -> None:
        self._poll_timer.pause()
