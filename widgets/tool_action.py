"""Collapsible tool call display — like toad's ToolCall widget."""

from textual.app import ComposeResult
from textual.containers import VerticalGroup
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

    def __init__(self, name: str, label: str) -> None:
        super().__init__()
        self.tool_name = name
        self.tool_label = label

    def compose(self) -> ComposeResult:
        icon = ICONS.get(self.tool_name, "🔧")
        # For Bash, just show the label (description). For others, show Name + label.
        if self.tool_name == "Bash":
            yield Static(f"{icon}  {self.tool_label}", classes="tool-header")
        elif self.tool_name == "Read":
            yield Static(f"{icon}  {self.tool_label}", classes="tool-header tool-dim")
        else:
            yield Static(f"{icon}  {self.tool_name} {self.tool_label}", classes="tool-header")
