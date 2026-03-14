"""
Unified TUI: quick-launch → optional wizard → experiment dashboard.

    uv run dashboard.py         # TUI with config memory
    uv run dashboard.py 10      # skip wizard, run 10 experiments
"""

import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import time

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from loop import DIR, MINS_PER_RUN, LoopReporter, run

RESULTS_PATH = os.path.join(DIR, "results.tsv")
RUN_LOG_PATH = os.path.join(DIR, "run.log")
CONFIG_PATH = os.path.join(DIR, ".autoresearch.json")
CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")

STEP_RE = re.compile(
    r"step\s+\d+\s+\(([\d.]+)%\)\s+\|\s+loss:\s+([\d.]+).*?"
    r"tok/sec:\s+([\d,]+).*?remaining:\s+(\d+)s"
)

DEFAULTS = {
    "model": "opus",
    "effort": "high",
    "num_runs": 10,
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
    try:
        out = subprocess.check_output(
            ["git", "branch", "--list", "autoresearch/*"], cwd=DIR, text=True,
        )
        return [b.strip().removeprefix("* ") for b in out.splitlines() if b.strip()]
    except Exception:
        return []


def get_results_summary():
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


def data_ok():
    return (
        os.path.isdir(os.path.join(CACHE_DIR, "data"))
        and os.path.isfile(os.path.join(CACHE_DIR, "tokenizer", "tokenizer.pkl"))
    )


def est_str(n):
    est = n * MINS_PER_RUN
    return f"~{est / 60:.1f}h" if est >= 60 else f"~{est}m"


# ── QuickLaunchScreen ─────────────────────────────────────────────────────────

class QuickLaunchScreen(Screen):
    """Compact launch screen: shows last config, editable run count, Start/Settings."""

    CSS = """
    QuickLaunchScreen { align: center middle; }
    #ql-box {
        width: 50;
        height: auto;
        border: round $primary;
        padding: 1 2;
    }
    #ql-box > Static { height: 1; }
    #ql-box > Label { height: 1; }
    #ql-title { text-style: bold; text-align: center; }
    #ql-config { color: $text-muted; }
    #ql-runs-input { width: 8; }
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
                    ago_str = f"{int(ago.total_seconds() / 60)}m ago"
                elif ago.total_seconds() < 86400:
                    ago_str = f"{ago.total_seconds() / 3600:.1f}h ago"
                else:
                    ago_str = f"{int(ago.days)}d ago"
                info += f" · last run {ago_str}"
            except ValueError:
                pass

        with Vertical(id="ql-box"):
            yield Static("autoresearch-mlx", id="ql-title")
            yield Static(info)
            yield Static(
                f"{self.cfg.get('model', 'opus')}/{self.cfg.get('effort', 'high')}",
                id="ql-config",
            )
            yield Label(f"Runs ({est_str(num)}):")
            yield Input(str(num), id="ql-runs-input", type="integer")
            yield Horizontal(
                Button("Start", variant="primary", id="ql-start"),
                Button("Settings", id="ql-settings"),
                id="ql-buttons",
            )

    def on_mount(self) -> None:
        self.query_one("#ql-runs-input", Input).focus()

    @on(Input.Changed, "#ql-runs-input")
    def update_label(self, event: Input.Changed):
        try:
            n = int(event.value)
            self.query_one("Label").update(f"Runs ({est_str(n)}):")
        except ValueError:
            pass

    @on(Input.Submitted, "#ql-runs-input")
    def on_enter(self, event: Input.Submitted):
        self.do_start()

    @on(Button.Pressed, "#ql-start")
    def do_start(self):
        try:
            self.cfg["num_runs"] = int(self.query_one("#ql-runs-input", Input).value)
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
            # Wizard was pushed on top of us — it already dismissed itself.
            # Replace ourselves with a fresh QuickLaunchScreen to reflect new config.
            self.app.switch_screen(QuickLaunchScreen(cfg))


# ── WizardScreen ──────────────────────────────────────────────────────────────

class WizardScreen(Screen):
    """Settings screen: branch, model, effort, API key."""

    CSS = """
    WizardScreen { align: center middle; }
    #wiz-box {
        width: 50;
        height: auto;
        border: round $primary;
        padding: 1 2;
    }
    #wiz-box > Static { height: 1; }
    #wiz-box > Label { height: 1; }
    .wiz-heading { text-style: bold; }
    #wiz-branch { width: 100%; }
    #wiz-new-tag { width: 30; display: none; }
    #wiz-model { width: 100%; }
    #wiz-effort { width: 100%; }
    #wiz-apikey { width: 100%; }
    #wiz-buttons { height: auto; align: center middle; margin-top: 1; }
    #wiz-buttons Button { margin: 0 1; min-width: 10; }
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)

    def compose(self) -> ComposeResult:
        branches = get_branches()
        current = get_branch()
        branch_opts = [(b, b) for b in branches] + [("+ New branch", "__new__")]
        default_b = self.cfg.get("branch", current)
        if default_b not in branches:
            default_b = current if current in branches else Select.BLANK

        models = [("opus", "opus"), ("sonnet", "sonnet"), ("haiku", "haiku")]
        efforts = [("high", "high"), ("medium", "medium"), ("low", "low")]

        with Vertical(id="wiz-box"):
            yield Static("Settings", classes="wiz-heading")

            # Data check
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
            yield Select(efforts, value=self.cfg.get("effort", "max"), id="wiz-effort")

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            yield Label("API Key (blank = use subscription):")
            yield Input(
                placeholder="sk-ant-...", password=True,
                id="wiz-apikey", value=api_key,
            )

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
        elif branch_val and branch_val != Select.BLANK:
            branch = str(branch_val)
            subprocess.run(["git", "checkout", branch], cwd=DIR, capture_output=True)
        else:
            branch = get_branch()

        model_val = self.query_one("#wiz-model", Select).value
        effort_val = self.query_one("#wiz-effort", Select).value

        cfg = {
            "model": str(model_val) if model_val != Select.BLANK else "opus",
            "effort": str(effort_val) if effort_val != Select.BLANK else "max",
            "num_runs": self.cfg.get("num_runs", 10),
            "branch": branch,
            "api_key": self.query_one("#wiz-apikey", Input).value.strip(),
        }
        save_config(cfg)
        self.dismiss(cfg)


# ── DashboardScreen ───────────────────────────────────────────────────────────

class DashboardReporter(LoopReporter):
    def __init__(self, screen: "DashboardScreen"):
        self.screen = screen

    def on_start(self, num_runs, model):
        self.screen.num_runs = num_runs
        self.screen.model_name = model

    def on_round_start(self, round_num, num_runs, elapsed_min, total_cost):
        self.screen.round_num = round_num
        self.screen.total_cost = total_cost
        self.screen.phase = "experimenting"
        self.screen._training_shown = False
        self.screen.query_one("#activity", RichLog).write(
            f"[bold]━━━ Round {round_num}/{num_runs} ━━━[/bold]"
        )

    def on_text(self, text):
        self.screen.query_one("#activity", RichLog).write(f"  [dim]{text[:200]}[/dim]")

    def on_tool_use(self, name, label):
        self.screen.query_one("#activity", RichLog).write(
            f"  [bold cyan]\\[{name}][/bold cyan] {label}"
        )

    def on_training_detected(self):
        if self.screen._training_shown:
            return
        self.screen._training_shown = True
        self.screen.phase = "training"
        self.screen.query_one("#train-panel").display = True
        self.screen.query_one("#train-bar", ProgressBar).update(total=100, progress=0)
        self.screen.query_one("#activity", RichLog).write(
            "  [bold green]> training (~5 min)...[/bold green]"
        )

    def on_round_done(self, round_cost, total_cost):
        self.screen.total_cost = total_cost
        self.screen.phase = "idle"
        self.screen.query_one("#train-panel").display = False
        note = " (included)" if round_cost == 0 else ""
        self.screen.query_one("#activity", RichLog).write(
            f"  [dim]── done (${round_cost:.2f}{note} | ${total_cost:.2f}) ──[/dim]"
        )
        self.screen.refresh_results()

    def on_round_failed(self, error):
        self.screen.phase = "error"
        self.screen.query_one("#train-panel").display = False
        self.screen.query_one("#activity", RichLog).write(
            f"  [bold red]── failed: {error} ──[/bold red]"
        )

    def on_finished(self, num_runs, elapsed_min, total_cost):
        self.screen.phase = "done"
        self.screen.query_one("#activity", RichLog).write(
            f"\n[bold]Done — {num_runs} rounds, {elapsed_min:.0f}m, ${total_cost:.2f}.[/bold]"
        )

    def on_stderr(self, line):
        self.screen.query_one("#activity", RichLog).write(
            f"  [bold red]SDK: {line}[/bold red]"
        )


class DashboardScreen(Screen):
    CSS = """
    DashboardScreen { layout: vertical; }
    #dash-header {
        dock: top; height: 1;
        background: $primary; color: $text; text-style: bold; padding: 0 1;
    }
    #dash-status {
        dock: top; height: 1;
        background: $panel; color: $text-muted; padding: 0 1;
    }
    #activity { height: 1fr; min-height: 6; }
    #train-panel {
        height: auto; max-height: 2;
        padding: 0 1; background: $surface; display: none;
    }
    #results { height: auto; max-height: 40%; border-top: heavy $primary; }
    """

    BINDINGS = [
        ("ctrl+c", "quit_app", "Stop"),
        ("ctrl+q", "quit_app", "Quit"),
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
        self._training_shown = False

    def compose(self) -> ComposeResult:
        yield Static(id="dash-header")
        yield Static(id="dash-status")
        yield RichLog(highlight=True, markup=True, id="activity")
        yield Horizontal(
            ProgressBar(total=100, show_eta=False, id="train-bar"),
            Static("", id="train-stats"),
            id="train-panel",
        )
        yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        self._update_header()
        self._update_status()
        self.refresh_results()
        self.set_interval(1.0, self._tick)
        self.run_loop()

    def _tick(self) -> None:
        self._update_status()
        if self.phase == "training":
            self._poll_run_log()

    def _update_header(self) -> None:
        effort = self.cfg.get("effort", "max")
        r = f"Round {self.round_num}/{self.num_runs}" if self.num_runs else "Starting"
        self.query_one("#dash-header", Static).update(
            f" {self._branch}  ·  {self.model_name}/{effort}  ·  {r}"
        )

    def _update_status(self) -> None:
        elapsed = (time.time() - self.start_time) / 60
        icon = {"starting": "○", "experimenting": "◉", "training": "▶",
                "idle": "○", "error": "✗", "done": "✓"}.get(self.phase, "○")
        cost = f"${self.total_cost:.2f}"
        if self.total_cost == 0 and self.round_num > 0:
            cost += " (included)"
        self.query_one("#dash-status", Static).update(
            f" {elapsed:.0f}m elapsed  ·  {cost}  ·  {icon} {self.phase}"
        )

    def _poll_run_log(self) -> None:
        if not os.path.exists(RUN_LOG_PATH):
            return
        try:
            with open(RUN_LOG_PATH, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 4096))
                tail = f.read().decode("utf-8", errors="replace")
            for seg in reversed(tail.replace("\r", "\n").split("\n")):
                m = STEP_RE.search(seg.strip())
                if m:
                    pct, loss, tps, rem = float(m.group(1)), m.group(2), m.group(3), m.group(4)
                    self.query_one("#train-bar", ProgressBar).update(total=100, progress=pct)
                    self.query_one("#train-stats", Static).update(
                        f" loss {loss}  ·  {tps} tok/s  ·  {rem}s left"
                    )
                    break
        except Exception:
            pass

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
        with open(RESULTS_PATH) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            return
        table.add_columns("", "val_bpb", "mem", "description")
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) >= 5:
                icon = {"keep": "[green]✓[/]", "discard": "[dim]✗[/]", "crash": "[red]![/]"}.get(cols[3], "?")
                table.add_row(icon, cols[1], cols[2], cols[4])

    def action_do_refresh(self) -> None:
        self.refresh_results()
    def action_quit_app(self) -> None:
        self.workers.cancel_all()
        self.app.exit()

    @work(exclusive=True)
    async def run_loop(self) -> None:
        import logging
        log = logging.getLogger("dashboard.run_loop")
        reporter = DashboardReporter(self)
        log.info("run_loop started, num_runs=%s", self.cfg.get("num_runs"))
        try:
            await run(self.cfg.get("num_runs", 10), reporter=reporter, config=self.cfg)
            log.info("run_loop completed normally")
        except asyncio.CancelledError:
            log.info("run_loop cancelled")
            try:
                self.query_one("#activity", RichLog).write("[dim]Stopped.[/dim]")
            except Exception:
                pass
        except Exception as e:
            log.exception("run_loop error")
            try:
                self.query_one("#activity", RichLog).write(
                    f"[bold red]Fatal: {type(e).__name__}: {e}[/bold red]"
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
        import logging
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
    import logging

    # File logger — flushes every line so `tail -f dashboard.log` works live
    handler = logging.FileHandler(LOG_FILE, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    # Force immediate flush on every log line
    handler.stream.reconfigure(line_buffering=True)
    log = logging.getLogger("dashboard")

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
        log.exception("Fatal error")
        raise
    finally:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            print(f"\n  Log written to {LOG_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
