"""Scrollable container that holds experiment rounds."""

from textual.app import ComposeResult
from textual.containers import VerticalGroup, VerticalScroll
from textual.widget import Widget


class ExperimentWindow(VerticalScroll):
    """Scrollable window holding all round widgets."""

    DEFAULT_CSS = """
    ExperimentWindow {
        height: 1fr;
        scrollbar-size-vertical: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield VerticalGroup(id="contents")

    async def post_widget(self, widget: Widget) -> Widget:
        """Mount a widget to the contents and scroll to show it."""
        await self.query_one("#contents").mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)
        return widget
