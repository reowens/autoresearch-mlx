"""Container for one experiment round — holds tool actions, training, results."""

from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.widget import Widget
from textual.widgets import Static


class RoundWidget(VerticalGroup):
    """One experiment round in the activity log."""

    def __init__(self, round_num: int, num_runs: int) -> None:
        super().__init__()
        self.round_num = round_num
        self.num_runs = num_runs

    def compose(self) -> ComposeResult:
        yield Static(
            f"── Round {self.round_num}/{self.num_runs} ──",
            classes="round-header",
            id="round-title",
        )

    def set_result(self, bpb: float, status: str, delta_str: str) -> None:
        """Update header with result and set border color."""
        icon = "✓" if status == "keep" else "✗"
        try:
            self.query_one("#round-title", Static).update(
                f"── Round {self.round_num}/{self.num_runs} ── {icon} {bpb:.4f} ({delta_str})"
            )
        except Exception:
            pass
        if status == "keep":
            self.set_class(True, "-keep")
        else:
            self.set_class(True, "-discard")

    async def mount(self, *widgets: Widget, **kwargs) -> None:
        """Mount child widget and scroll parent window to bottom."""
        await super().mount(*widgets, **kwargs)
        try:
            from widgets.experiment_window import ExperimentWindow
            window = self.query_ancestor(ExperimentWindow)
            window.call_after_refresh(window.scroll_end, animate=False)
        except Exception:
            pass
