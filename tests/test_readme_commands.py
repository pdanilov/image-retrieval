"""Every `cbir` command in the README has to parse.

The README documented `--sweep-k` for some time after the flag became `--sweep`; seven
copy-pasted commands failed at argument parsing, and nothing noticed because nothing
ran them. This runs each one as far as the parser and no further -- the work itself is
stubbed, so the suite stays fast and touches no data.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

import cbir.cli as cli

README = Path(__file__).resolve().parents[1] / "README.md"
COMMAND = re.compile(r"^\s*(?:#\s*or:\s*)?uv run (cbir .*?)$", re.MULTILINE)


def commands() -> list[str]:
    """Every `uv run cbir ...` line in the README, line continuations joined."""
    text = README.read_text()
    # Join `\`-continued lines first, or the second half parses as its own command.
    text = re.sub(r"\\\n\s*", " ", text)
    # A trailing `# ...` is prose in the code block, not an argument.
    return [match.group(1).split("#")[0].strip() for match in COMMAND.finditer(text)]


def test_the_readme_actually_contains_commands():
    # Guards the regex: if it stops matching, every other test here passes vacuously.
    found = commands()
    assert len(found) >= 8, found


@pytest.fixture
def parsed(monkeypatch):
    """Run the CLI up to the parser, recording what it built."""
    seen: list = []

    monkeypatch.setattr("cbir.eval.runner.run_all", lambda configs, **kwargs: seen.extend(configs) or [])
    monkeypatch.setattr("cbir.data.revisitop.download", lambda name: seen.append(("download", name)))
    monkeypatch.setattr("cbir.eval.tracking.replay", lambda records: 0)
    # Training and whitening are hours of GPU each. Unstubbed, documenting a `cbir train`
    # line in the README makes this suite start a real 100-epoch run and hang -- which is
    # exactly what happened, and is why `test_every_documented_subcommand_is_stubbed`
    # below fails on a new subcommand rather than waiting to find out.
    monkeypatch.setattr("cbir.training.loop.train", lambda config, **kwargs: Path("stub.pth"))
    monkeypatch.setattr("cbir.training.whitening.fit", lambda *args, **kwargs: {})
    monkeypatch.setattr("cbir.training.whitening.attach", lambda *args, **kwargs: None)
    return seen


STUBBED = {"download", "evaluate", "results", "track", "train", "whiten"}
"""Subcommands whose real work the `parsed` fixture replaces.

`results` does no work worth stubbing -- it reads the record store -- but it is listed so
this set reads as "every subcommand", which is what the guard below checks."""


def test_every_documented_subcommand_is_stubbed():
    """A README command whose work is not stubbed does not fail here -- it *runs*."""
    used = {command.split()[1] for command in commands()}
    assert used <= STUBBED, f"unstubbed subcommand(s) in the README: {sorted(used - STUBBED)}"


@pytest.mark.parametrize("command", commands(), ids=lambda c: c[:60])
def test_readme_command_parses(command, parsed, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", shlex.split(command))

    try:
        cli.main()
    except SystemExit as exit_code:  # tyro exits non-zero on a parse error
        if exit_code.code:
            pytest.fail(f"{command!r} failed to parse:\n{capsys.readouterr().err or capsys.readouterr().out}")
