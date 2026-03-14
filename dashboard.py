"""
Unified TUI: quick-launch → optional wizard → experiment dashboard.

    uv run dashboard.py         # TUI with config memory
    uv run dashboard.py 10      # skip wizard, run 10 experiments
"""

import asyncio
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import time

from textual import on, work
from textual.app import App, ComposeResult
from textual.color import Gradient
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    Select,
    Static,
)

from loop import DIR, MINS_PER_RUN, LoopReporter, run

RESULTS_PATH = os.path.join(DIR, "results.tsv")
RUN_LOG_PATH = os.path.join(DIR, "run.log")
CONFIG_PATH = os.path.join(DIR, ".autoresearch.json")
CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")

DEFAULTS = {
    "model": "opus",
    "effort": "medium",
    "num_runs": 10,
    "time_budget": 5,
    "branch": "",
    "api_key": "",
}


# ── Config persistence ────────────────────────────────────────────────────────

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return {**DEFAULTS, **json.load(f)}
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_config(cfg):
    out = {k: v for k, v in cfg.items() if k != "api_key"}
    out["api_key_set"] = bool(cfg.get("api_key"))
    out["last_run"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(CONFIG_PATH, "w") as f:
        json.dump(out, f, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_branch():
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=DIR, text=True,
        ).strip()
    except Exception:
        return "?"


def get_branches():
    current = get_branch()
    branches = []
    if current and current != "?":
        branches.append(current)
    try:
        out = subprocess.check_output(
            ["git", "branch", "--list", "autoresearch/*", "experiments"], cwd=DIR, text=True,
        )
        for b in out.splitlines():
            name = b.strip().removeprefix("* ")
            if name and name not in branches:
                branches.append(name)
    except Exception:
        pass
    return branches


def get_results_summary():
    try:
        if not os.path.exists(RESULTS_PATH) or os.path.getsize(RESULTS_PATH) < 50:
            return None
        with open(RESULTS_PATH) as f:
            lines = [l for l in f.readlines() if l.strip()]
        if len(lines) < 2:
            return None
        n = len(lines) - 1
        kept = [l for l in lines[1:] if "\tkeep\t" in l]
        best_bpb = min((float(l.split("\t")[1]) for l in kept), default=None)
        return {"total": n, "kept": len(kept), "best_bpb": best_bpb}
    except Exception:
        return None


def data_ok():
    return (
        os.path.isdir(os.path.join(CACHE_DIR, "data"))
        and os.path.isfile(os.path.join(CACHE_DIR, "tokenizer", "tokenizer.pkl"))
    )


def est_str(n, time_budget=5):
    mins_per = time_budget + 2
    est = n * mins_per
    return f"~{est / 60:.1f}h" if est >= 60 else f"~{est}m"


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ── QuickLaunchScreen ─────────────────────────────────────────────────────────

class QuickLaunchScreen(Screen):
    CSS = """
    QuickLaunchScreen { align: center middle; }
    #ql-box {
        width: 50; height: auto;
        border: round $primary; padding: 1 2;
    }
    #ql-box > Static { height: 1; }
    #ql-box > Label { height: 1; }
    #ql-title { text-style: bold; text-align: center; }
    #ql-config { color: $text-muted; }
    #ql-runs-input { width: 8; }
    #ql-time-input { width: 8; }
    #ql-buttons { height: auto; align: center middle; margin-top: 1; }
    #ql-buttons Button { margin: 0 1; min-width: 12; }
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        branch = get_branch()
        results = get_results_summary()
        num = int(self.cfg.get("num_runs", 10))
        tb = int(self.cfg.get("time_budget", 5))
        effort = self.cfg.get("effort", "medium")

        info = branch
        if results:
            info += f" · {results['total']} exp ({results['kept']} kept)"
            if results["best_bpb"]:
                info += f" · best {results['best_bpb']:.4f}"
        last = self.cfg.get("last_run", "")
        if last:
            try:
                dt = datetime.datetime.fromisoformat(last)
                ago = datetime.datetime.now() - dt
                if ago.total_seconds() < 3600:
                    info += f" · {int(ago.total_seconds() / 60)}m ago"
                elif ago.total_seconds() < 86400:
                    info += f" · {ago.total_seconds() / 3600:.1f}h ago"
                else:
                    info += f" · {int(ago.days)}d ago"
            except ValueError:
                pass

        with Vertical(id="ql-box"):
            yield Static("autoresearch-mlx", id="ql-title")
            yield Static(info)
            yield Static(f"{self.cfg.get('model', 'opus')}/{effort}", id="ql-config")
            yield Label(f"Runs ({est_str(num, tb)}):", id="ql-runs-label")
            yield Input(str(num), id="ql-runs-input", type="integer")
            yield Label(f"Training time: {tb} min/run", id="ql-time-label")
            yield Input(str(tb), id="ql-time-input", type="integer")
            yield Horizontal(
                Button("Start", variant="primary", id="ql-start"),
                Button("Settings", id="ql-settings"),
                id="ql-buttons",
            )

    def on_mount(self) -> None:
        self.query_one("#ql-runs-input", Input).focus()

    def _update_estimates(self):
        try:
            n = int(self.query_one("#ql-runs-input", Input).value)
            tb = int(self.query_one("#ql-time-input", Input).value)
            self.query_one("#ql-runs-label", Label).update(f"Runs ({est_str(n, tb)}):")
            self.query_one("#ql-time-label", Label).update(f"Training time: {tb} min/run")
        except (ValueError, Exception):
            pass

    @on(Input.Changed, "#ql-runs-input")
    def on_runs_changed(self, event: Input.Changed):
        self._update_estimates()

    @on(Input.Changed, "#ql-time-input")
    def on_time_changed(self, event: Input.Changed):
        self._update_estimates()

    @on(Input.Submitted)
    def on_enter(self, event: Input.Submitted):
        self.do_start()

    @on(Button.Pressed, "#ql-start")
    def do_start(self):
        try:
            self.cfg["num_runs"] = int(self.query_one("#ql-runs-input", Input).value)
        except ValueError:
            pass
        try:
            self.cfg["time_budget"] = int(self.query_one("#ql-time-input", Input).value)
        except ValueError:
            pass
        self.cfg["branch"] = get_branch()
        save_config(self.cfg)
        self.app.push_screen(DashboardScreen(self.cfg))

    @on(Button.Pressed, "#ql-settings")
    def do_settings(self):
        self.app.push_screen(WizardScreen(self.cfg), callback=self._wizard_done)

    def _wizard_done(self, cfg):
        if cfg:
            self.cfg = cfg
            self.app.switch_screen(QuickLaunchScreen(cfg))


# ── WizardScreen ──────────────────────────────────────────────────────────────

class WizardScreen(Screen):
    CSS = """
    WizardScreen { align: center middle; }
    #wiz-box {
        width: 50; height: auto;
        border: round $primary; padding: 1 2;
    }
    #wiz-box > Static { height: 1; }
    #wiz-box > Label { height: 1; }
    .wiz-heading { text-style: bold; }
    #wiz-branch { width: 100%; }
    #wiz-new-tag { width: 30; display: none; }
    #wiz-model { width: 100%; }
    #wiz-effort { width: 100%; }
    #wiz-time { width: 100%; }
    #wiz-apikey { width: 100%; }
    #wiz-buttons { height: auto; align: center middle; margin-top: 1; }
    #wiz-buttons Button { margin: 0 1; min-width: 10; }
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)

    def compose(self) -> ComposeResult:
        branches = get_branches()
        default_b = self.cfg.get("branch") or (branches[0] if branches else "__new__")
        if default_b not in branches and default_b != "__new__":
            default_b = branches[0] if branches else "__new__"
        branch_opts = [(b, b) for b in branches] + [("+ New branch", "__new__")]
        models = [("opus", "opus"), ("sonnet", "sonnet"), ("haiku", "haiku")]
        efforts = [("medium", "medium"), ("high", "high"), ("low", "low")]

        with Vertical(id="wiz-box"):
            yield Static("Settings", classes="wiz-heading")
            if data_ok():
                shards = len([f for f in os.listdir(os.path.join(CACHE_DIR, "data"))
                              if f.endswith(".parquet")])
                yield Static(f"Data: [green]✓[/] {shards} shards", markup=True)
            else:
                yield Static("Data: [red]✗[/] not found — run prepare first", markup=True)
            yield Label("Branch:")
            yield Select(branch_opts, value=default_b, id="wiz-branch")
            yield Input(placeholder="tag (e.g. mar14)", id="wiz-new-tag")
            yield Label("Model:")
            yield Select(models, value=self.cfg.get("model", "opus"), id="wiz-model")
            yield Label("Effort:")
            yield Select(efforts, value=self.cfg.get("effort", "medium"), id="wiz-effort")
            times = [("5 min", 5), ("10 min", 10), ("15 min", 15), ("20 min", 20)]
            yield Label("Training time per run:")
            yield Select(times, value=self.cfg.get("time_budget", 5), id="wiz-time")
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            yield Label("API Key (blank = subscription):")
            yield Input(placeholder="sk-ant-...", password=True, id="wiz-apikey", value=api_key)
            yield Horizontal(
                Button("Back", id="wiz-back"),
                Button("Save", variant="primary", id="wiz-save"),
                id="wiz-buttons",
            )

    @on(Select.Changed, "#wiz-branch")
    def branch_changed(self, event: Select.Changed):
        tag = self.query_one("#wiz-new-tag", Input)
        tag.display = event.value == "__new__"
        if tag.display:
            tag.focus()

    @on(Button.Pressed, "#wiz-back")
    def go_back(self):
        self.dismiss(None)

    @on(Input.Submitted)
    def on_enter(self, event: Input.Submitted):
        self.do_save()

    @on(Button.Pressed, "#wiz-save")
    def do_save(self):
        branch_val = self.query_one("#wiz-branch", Select).value
        if branch_val == "__new__":
            tag = self.query_one("#wiz-new-tag", Input).value.strip()
            if tag:
                branch = f"autoresearch/{tag}"
                subprocess.run(["git", "checkout", "-b", branch], cwd=DIR, capture_output=True)
            else:
                branch = get_branch()
        elif branch_val:
            branch = str(branch_val)
            subprocess.run(["git", "checkout", branch], cwd=DIR, capture_output=True)
        else:
            branch = get_branch()
        model_val = self.query_one("#wiz-model", Select).value
        effort_val = self.query_one("#wiz-effort", Select).value
        time_val = self.query_one("#wiz-time", Select).value
        cfg = {
            "model": str(model_val) if model_val else "opus",
            "effort": str(effort_val) if effort_val else "medium",
            "num_runs": self.cfg.get("num_runs", 10),
            "time_budget": int(time_val) if time_val else 5,
            "branch": branch,
            "api_key": self.query_one("#wiz-apikey", Input).value.strip(),
        }
        save_config(cfg)
        self.dismiss(cfg)


# ── DashboardScreen ───────────────────────────────────────────────────────────

from widgets.round_widget import RoundWidget
from widgets.tool_action import ToolAction
from widgets.training_progress import TrainingProgress
from widgets.experiment_window import ExperimentWindow
from widgets.thinking_indicator import ThinkingIndicator


class DashboardReporter(LoopReporter):
    """Mounts real widgets instead of writing to RichLog."""

    def __init__(self, screen: "DashboardScreen"):
        self.screen = screen
        self._read_buffer = []
        self._seen_reads = set()  # deduplicate reads within a round
        self._text_buf = ""
        self._current_round = None
        self._training_widget = None
        self._thinking_widget = None
        self._round_summaries = []

    def _touch(self):
        self.screen._last_msg_time = time.time()
        self.screen._thinking_dots = 0
        # Remove thinking indicator when activity resumes
        if self.screen._thinking_indicator and self.screen._thinking_indicator.is_attached:
            self.screen._thinking_indicator.remove()
            self.screen._thinking_indicator = None

    async def _flush_reads(self):
        if self._read_buffer and self._current_round:
            # Deduplicate: skip files already read this round
            new_files = [f for f in self._read_buffer if f not in self._seen_reads]
            self._seen_reads.update(self._read_buffer)
            self._read_buffer = []
            if new_files:
                files = ", ".join(new_files)
                await self._current_round.mount(ToolAction("Read", files))

    def _classify_line(self, line):
        """Return CSS class for a line of agent text."""
        low = line.lower()
        if any(kw in low for kw in ["val_bpb", "improved", "keep", "discard", "result", "better", "worse"]):
            return "thinking-result"
        if line.startswith("- "):
            return "thinking-bullet"
        return "thinking-text"

    async def _flush_text(self):
        if self._text_buf.strip() and self._current_round:
            text = self._clean_markdown(self._text_buf.strip())
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    css = self._classify_line(line)
                    await self._current_round.mount(
                        Static(line, classes=css)
                    )
            self._text_buf = ""

    async def on_start(self, num_runs, model):
        self.screen.num_runs = num_runs
        self.screen.model_name = model

    async def on_round_start(self, round_num, num_runs, elapsed_min, total_cost):
        self._touch()
        await self._flush_reads()
        await self._flush_text()
        self.screen.round_num = round_num
        self.screen.total_cost = total_cost
        self.screen.phase = "experimenting"
        self._training_widget = None
        self._thinking_widget = None
        self._seen_reads = set()
        self._current_round = RoundWidget(round_num, num_runs)
        await self.screen.query_one(ExperimentWindow).post_widget(self._current_round)
        self.screen._update_session_bar()

    def _clean_markdown(self, text):
        """Strip markdown formatting for plain text display."""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        return text

    async def on_text(self, text):
        self._touch()
        await self._flush_reads()
        if not self._text_buf and self._current_round:
            text = self._clean_markdown(text)
            for line in text.strip().split("\n")[:3]:
                line = line.strip()
                if line:
                    css = self._classify_line(line)
                    await self._current_round.mount(
                        Static(line, classes=css)
                    )

    async def on_text_delta(self, chunk):
        self._touch()
        self._text_buf += chunk
        while "\n" in self._text_buf:
            line, self._text_buf = self._text_buf.split("\n", 1)
            line = self._clean_markdown(line.strip())
            if line and self._current_round:
                css = self._classify_line(line)
                await self._current_round.mount(
                    Static(line, classes=css)
                )

    async def on_tool_start(self, name):
        self._touch()
        await self._flush_reads()
        await self._flush_text()

    async def on_tool_use(self, name, label):
        self._touch()
        if name == "Read":
            self._read_buffer.append(label)
            return
        await self._flush_reads()
        await self._flush_text()
        # Only end training phase on Bash commands (actual experiment actions)
        # SDK-internal tools (ToolSearch, TaskOutput) can fire during training
        if self._training_widget and self.screen.phase == "training" and name == "Bash":
            self.screen.phase = "experimenting"
            self._training_widget.stop()
        if not self._current_round:
            return
        if name == "Bash" and "experiment:" in str(label):
            desc = str(label).split("experiment:")[-1].strip().rstrip('"').rstrip("'")
            if desc:
                self._current_round.set_description(desc)
        await self._current_round.mount(ToolAction(name, label))

    async def on_training_detected(self):
        self._touch()
        await self._flush_reads()
        await self._flush_text()
        if self._training_widget:
            return
        self.screen.phase = "training"
        self._training_widget = TrainingProgress(RUN_LOG_PATH)
        if self._current_round:
            await self._current_round.mount(self._training_widget)

    async def on_round_done(self, round_cost, total_cost, usage=None):
        self._touch()
        await self._flush_reads()
        await self._flush_text()
        self.screen.total_cost = total_cost
        if usage:
            self.screen._total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        self.screen.phase = "idle"
        if self._training_widget:
            self._training_widget.stop()
            self._training_widget = None
        try:
            with open(RESULTS_PATH) as f:
                last = f.readlines()[-1].strip().split("\t")
            bpb = float(last[1])
            status = last[3]
            results = get_results_summary()
            best = results["best_bpb"] if results else None
            if best and self._current_round:
                delta = bpb - best
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                icon = "✓" if status == "keep" else "✗"
                css = "result-keep" if status == "keep" else (
                    "result-near-miss" if delta < 0.02 else "result-discard"
                )
                await self._current_round.mount(
                    Static(f"{icon} {bpb:.4f} ({delta_str} vs best)", classes=css)
                )
                self._current_round.set_result(bpb, status, delta_str)
                self._round_summaries.append(f"R{self.screen.round_num}: {icon} {bpb:.4f}")
        except Exception:
            pass
        self.screen._update_session_bar()
        self.screen.refresh_results()
        # Update best in header
        results = get_results_summary()
        if results and results["best_bpb"]:
            self.screen._best_bpb = results["best_bpb"]

    async def on_round_failed(self, error):
        self._touch()
        await self._flush_reads()
        await self._flush_text()
        self.screen.phase = "error"
        if self._training_widget:
            self._training_widget.stop()
            self._training_widget = None
        if self._current_round:
            await self._current_round.mount(
                Static(f"✗ {error}", classes="result-discard")
            )
        self._round_summaries.append(f"R{self.screen.round_num}: ✗")

    async def on_finished(self, num_runs, elapsed_min, total_cost, total_usage=None):
        await self._flush_reads()
        await self._flush_text()
        self.screen.phase = "done"
        self.screen._update_header()
        self.screen._update_session_bar()
        window = self.screen.query_one(ExperimentWindow)
        if self._round_summaries:
            summary = " | ".join(self._round_summaries)
            await window.post_widget(Static(summary, classes="round-header"))
        tokens = f" · {self.screen._total_tokens // 1000}k tok" if self.screen._total_tokens > 0 else ""
        await window.post_widget(
            Static(f"Done — {num_runs} rounds · {elapsed_min:.0f}m · ${total_cost:.2f}{tokens}",
                   classes="round-header")
        )
        self.screen.app.bell()

    async def on_stderr(self, line):
        pass  # Logged to dashboard.log


class DashboardScreen(Screen):
    CSS_PATH = "dashboard.tcss"

    BINDINGS = [
        ("ctrl+c", "quit_app", "Stop"),
        ("ctrl+q", "quit_app", "Quit"),
        ("ctrl+r", "restart", "Restart"),
        ("r", "do_refresh", "Refresh"),
    ]

    round_num: reactive[int] = reactive(0)
    num_runs: reactive[int] = reactive(0)
    total_cost: reactive[float] = reactive(0.0)
    phase: reactive[str] = reactive("starting")
    model_name: reactive[str] = reactive("opus")

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.start_time = time.time()
        self._branch = get_branch()
        self._last_msg_time = time.time()
        self._thinking_dots = 0
        self._total_tokens = 0
        self._thinking_indicator = None
        self._reporter = None
        self._best_bpb = None

    def compose(self) -> ComposeResult:
        yield Static(id="status-line")
        yield ExperimentWindow()
        yield Static(id="session-bar", markup=True)
        yield DataTable(id="results", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        results = get_results_summary()
        if results and results["best_bpb"]:
            self._best_bpb = results["best_bpb"]
        self._update_header()
        self._update_session_bar()
        self.refresh_results()
        self.set_interval(1.0, self._tick)
        self.run_loop()

    def _tick(self) -> None:
        if self.phase == "done":
            return
        self._update_header()
        # Show thinking indicator after 5s of silence
        thinking_secs = time.time() - self._last_msg_time
        if self.phase == "experimenting" and thinking_secs > 5 and not self._thinking_indicator:
            if self._reporter and self._reporter._current_round:
                indicator = ThinkingIndicator()
                self._thinking_indicator = indicator
                self.call_later(self._mount_thinking, indicator)

    async def _mount_thinking(self, indicator: "ThinkingIndicator") -> None:
        if self._reporter and self._reporter._current_round and indicator.is_attached is False:
            await self._reporter._current_round.mount(indicator)

    def _update_header(self) -> None:
        effort = self.cfg.get("effort", "medium")
        elapsed = fmt_elapsed(time.time() - self.start_time)
        r = f"Round {self.round_num}/{self.num_runs}" if self.num_runs else "Starting"
        thinking_secs = time.time() - self._last_msg_time
        if self.phase == "experimenting" and thinking_secs > 3:
            self._thinking_dots = (self._thinking_dots % 3) + 1
            dots = "·" * self._thinking_dots + " " * (3 - self._thinking_dots)
            suffix = f"  thinking {dots} ({int(thinking_secs)}s)"
        elif self.phase == "training":
            suffix = "  ▶ Training"
        elif self.phase == "done":
            suffix = "  ✓ done"
        elif self.phase == "error":
            suffix = "  ✗ error"
        else:
            suffix = ""
        best = f"  ·  best: {self._best_bpb:.4f}" if self._best_bpb else ""
        self.query_one("#status-line", Static).update(
            f" {self._branch}  ·  {self.model_name}/{effort}  ·  {r}  ·  {elapsed}{best}{suffix}"
        )

    def _update_session_bar(self) -> None:
        r = f"{self.round_num}/{self.num_runs}" if self.num_runs else "0/0"
        cost = f"${self.total_cost:.2f}"
        if self.total_cost == 0 and self.round_num > 0:
            cost += " (incl)"
        tokens = f" · {self._total_tokens // 1000}k tok" if self._total_tokens > 0 else ""
        if self.num_runs:
            bw = 15
            filled = int(bw * self.round_num / self.num_runs)
            bar = f"[green]{'█' * filled}[/][dim]{'░' * (bw - filled)}[/]"
        else:
            bar = f"[dim]{'░' * 15}[/]"
        self.query_one("#session-bar", Static).update(f" {bar} {r} · {cost}{tokens}")

    def watch_round_num(self) -> None:
        self._update_header()
    def watch_num_runs(self) -> None:
        self._update_header()
    def watch_model_name(self) -> None:
        self._update_header()

    def refresh_results(self) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=True)
        if not os.path.exists(RESULTS_PATH):
            return
        try:
            with open(RESULTS_PATH) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) < 2:
                return
            table.add_columns("", "val_bpb", "delta", "description")
            data_lines = [l for l in lines[1:] if len(l.split("\t")) >= 5]
            rows = [l.split("\t") for l in data_lines]
            kept = [r for r in rows if r[3] == "keep"]
            best_bpb = min((float(r[1]) for r in kept), default=None) if kept else None
            baseline = rows[0] if rows and rows[0][3] == "keep" else None
            best = next((r for r in kept if best_bpb and float(r[1]) == best_bpb), None)
            shown = set()
            sorted_rows = []
            for r in [best, baseline]:
                if r and id(r) not in shown:
                    sorted_rows.append(r)
                    shown.add(id(r))
            for r in reversed(rows):
                if id(r) not in shown:
                    sorted_rows.append(r)
                    shown.add(id(r))
            max_rows = 8
            for i, cols in enumerate(sorted_rows):
                if i >= max_rows:
                    remaining = len(sorted_rows) - max_rows
                    table.add_row("", "", "", f"... and {remaining} more")
                    break
                status, bpb = cols[3], cols[1]
                is_best = best_bpb and status == "keep" and float(bpb) == best_bpb
                delta = float(bpb) - best_bpb if best_bpb else 999
                if is_best:
                    icon = "[bold green]★[/]"
                    delta_str = "best"
                elif status == "keep":
                    icon = "[green]✓[/]"
                    delta_str = f"+{delta:.4f}" if delta > 0 else "baseline"
                elif status == "crash" or float(bpb) == 0:
                    icon = "[red]![/]"
                    delta_str = "crash"
                elif delta > 1.0:
                    icon = "[red]![/]"
                    delta_str = f"+{delta:.3f}"
                elif delta < 0.02:
                    icon = "[yellow]~[/]"
                    delta_str = f"+{delta:.4f}"
                else:
                    icon = "[dim]✗[/]"
                    delta_str = f"+{delta:.4f}"
                table.add_row(icon, bpb, delta_str, cols[4])
        except Exception:
            pass

    def action_do_refresh(self) -> None:
        self.refresh_results()

    def action_quit_app(self) -> None:
        self.workers.cancel_all()
        self.app.exit()

    def action_restart(self) -> None:
        self.workers.cancel_all()
        subprocess.run(["git", "checkout", "--", "train.py"], cwd=DIR, capture_output=True)
        self.app.switch_screen(QuickLaunchScreen(self.cfg))

    @work(exclusive=True)
    async def run_loop(self) -> None:
        log = logging.getLogger("dashboard.run_loop")
        reporter = DashboardReporter(self)
        self._reporter = reporter
        log.info("run_loop started, num_runs=%s", self.cfg.get("num_runs"))
        try:
            await run(self.cfg.get("num_runs", 10), reporter=reporter, config=self.cfg)
            log.info("run_loop completed normally")
        except asyncio.CancelledError:
            log.info("run_loop cancelled")
        except Exception as e:
            log.exception("run_loop error")
            try:
                window = self.query_one(ExperimentWindow)
                await window.post_widget(
                    Static(f"Fatal: {type(e).__name__}: {e}", classes="result-discard")
                )
            except Exception:
                pass


# ── App ───────────────────────────────────────────────────────────────────────

class AutoresearchApp(App):
    CSS = "Screen { align: center middle; }"
    BINDINGS = [("ctrl+c", "quit", "Stop"), ("ctrl+q", "quit", "Quit")]

    def __init__(self, num_runs_override: int | None = None):
        super().__init__()
        self.num_runs_override = num_runs_override

    def on_mount(self) -> None:
        log = logging.getLogger("dashboard.app")
        cfg = load_config()
        log.info("Config loaded: %s, pushing: %s",
                 cfg is not None,
                 "Dashboard" if self.num_runs_override else ("QuickLaunch" if cfg else "Wizard"))
        if self.num_runs_override:
            if cfg is None:
                cfg = dict(DEFAULTS)
            cfg["num_runs"] = self.num_runs_override
            cfg["branch"] = get_branch()
            save_config(cfg)
            self.push_screen(DashboardScreen(cfg))
        elif cfg:
            self.push_screen(QuickLaunchScreen(cfg))
        else:
            default_cfg = dict(DEFAULTS)
            default_cfg["branch"] = get_branch()
            self.push_screen(WizardScreen(default_cfg), callback=self._wizard_done)

    def _wizard_done(self, cfg):
        if cfg:
            save_config(cfg)
            self.push_screen(QuickLaunchScreen(cfg))
        else:
            self.exit()


# ── Entry point ───────────────────────────────────────────────────────────────

LOG_FILE = os.path.join(DIR, "dashboard.log")


def main():
    handler = logging.FileHandler(LOG_FILE, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        handler.stream.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    num_runs = None
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
            if num_runs <= 0:
                raise ValueError("must be positive")
        except ValueError as e:
            sys.exit(f"  Error: {e}")

    try:
        AutoresearchApp(num_runs_override=num_runs).run()
    except Exception:
        logging.getLogger("dashboard").exception("Fatal error")
        raise
    finally:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            print(f"\n  Log: {LOG_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
