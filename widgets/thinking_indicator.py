"""Animated thinking indicator — shows dots while agent reasons."""

from time import monotonic

from rich.text import Text
from textual.widget import Widget


class ThinkingIndicator(Widget):
    """Animated ··· indicator mounted in the round while agent thinks."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        margin: 0 0 0 1;
        color: $text-muted;
    }
    """

    def on_mount(self) -> None:
        self._start = monotonic()
        self.auto_refresh = 1 / 2  # 2 fps

    def render(self) -> Text:
        elapsed = int(monotonic() - self._start)
        dots = "·" * ((elapsed % 3) + 1)
        return Text(f"thinking {dots} ({elapsed}s)", style="dim italic")
