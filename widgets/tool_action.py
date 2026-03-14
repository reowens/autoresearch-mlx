"""Collapsible tool call display — like toad's ToolCall widget."""

from textual import on, events
from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.reactive import var
from textual.widgets import Static

ICONS = {
    "Read": "📖",
    "Edit": "✏️",
    "Write": "📝",
    "Bash": "⚡",
    "Glob": "🔍",
    "Grep": "🔍",
}


class ToolAction(VerticalGroup):
    """A single tool call displayed in the activity log."""

    expanded: var[bool] = var(False, toggle_class="-expanded")

    def __init__(self, name: str, label: str) -> None:
        super().__init__()
        self.tool_name = name
        self.tool_label = label

    def compose(self) -> ComposeResult:
        icon = ICONS.get(self.tool_name, "🔧")

        if self.tool_name == "Read" and ", " in self.tool_label:
            # Collapsible: show file count, expand to show full list
            count = len(self.tool_label.split(", "))
            yield Static(f"{icon}  {count} files", classes="tool-header tool-dim", id="tool-summary")
            yield Static(f"{icon}  {self.tool_label}", classes="tool-header tool-dim", id="tool-detail")
        elif self.tool_name == "Read":
            yield Static(f"{icon}  {self.tool_label}", classes="tool-header tool-dim")
        elif self.tool_name == "Bash":
            yield Static(f"{icon}  {self.tool_label}", classes="tool-header")
        elif self.tool_name in ("Edit", "Write"):
            yield Static(f"{icon}  {self.tool_name} {self.tool_label}", classes="tool-header")
        else:
            yield Static(f"{icon}  {self.tool_name} {self.tool_label}", classes="tool-header tool-dim")

    @on(events.Click, "#tool-summary")
    def on_click_summary(self, event: events.Click) -> None:
        event.stop()
        self.expanded = True

    @on(events.Click, "#tool-detail")
    def on_click_detail(self, event: events.Click) -> None:
        event.stop()
        self.expanded = False
