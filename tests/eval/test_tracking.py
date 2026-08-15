import sys

import pytest

from cbir.eval.results import RunRecord
from cbir.eval.tracking import config_of, flatten_metrics, replay, run_name, track


def _record(**kwargs) -> RunRecord:
    defaults = {
        "dataset": "roxford5k",
        "held_out_dataset": "rparis6k",
        "technique": "bow",
        "params": {"k": 5000, "seed": 0},
        "metrics": {
            "easy": {"map": 0.1775, "mp_at_k": 0.3211, "k": 10, "num_queries": 68, "num_excluded": 2},
            "medium": {"map": 0.1484, "mp_at_k": 0.3050, "k": 10, "num_queries": 70, "num_excluded": 0},
        },
        "seconds": 2344.6,
        "commit": "abc1234",
    }
    return RunRecord(**{**defaults, **kwargs})


class _FakeTrackio:
    def __init__(self):
        self.calls = []

    def init(self, project, name, config):
        self.calls.append(("init", project, name, config))

    def log(self, metrics):
        self.calls.append(("log", metrics))

    def finish(self):
        self.calls.append(("finish",))


@pytest.fixture
def fake_trackio(monkeypatch):
    module = _FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", module)
    return module


def test_flatten_metrics_namespaces_by_protocol():
    flat = flatten_metrics(_record())

    assert flat["easy/map"] == pytest.approx(0.1775)
    assert flat["medium/map"] == pytest.approx(0.1484)
    # num_queries/num_excluded travel too: a protocol scored over fewer queries is not
    # comparable to one scored over all of them, and the dashboard must be able to say so.
    assert flat["easy/num_excluded"] == 2
    assert flat["seconds"] == pytest.approx(2344.6)


def test_flatten_metrics_skips_non_numeric_values():
    flat = flatten_metrics(_record(metrics={"easy": {"map": 0.5, "note": "rerun"}}))
    assert flat == {"easy/map": 0.5, "seconds": pytest.approx(2344.6)}


def test_flatten_metrics_omits_seconds_when_untimed():
    assert "seconds" not in flatten_metrics(_record(seconds=None))


def test_run_name_identifies_the_configuration():
    assert run_name(_record()) == "bow-roxford5k-k5000-seed0"


def test_config_carries_the_held_out_dataset_and_commit():
    config = config_of(_record())
    assert config["held_out_dataset"] == "rparis6k"
    assert config["commit"] == "abc1234"
    assert config["k"] == 5000


def test_track_logs_once_and_always_finishes(fake_trackio):
    track(_record())

    kinds = [call[0] for call in fake_trackio.calls]
    assert kinds == ["init", "log", "finish"]


def test_track_finishes_even_when_logging_raises(monkeypatch, fake_trackio):
    def boom(metrics):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fake_trackio, "log", boom)

    with pytest.raises(RuntimeError):
        track(_record())
    # An unfinished run leaves the project's SQLite row open; the next `trackio init`
    # would attach to it and mix two runs' scalars together.
    assert fake_trackio.calls[-1] == ("finish",)


def test_track_is_a_no_op_without_trackio_installed(monkeypatch):
    # trackio is an optional dependency: evaluation on a machine without it must
    # still complete and still write results/runs.jsonl.
    monkeypatch.setitem(sys.modules, "trackio", None)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def no_trackio(name, *args, **kwargs):
        if name == "trackio":
            raise ImportError("No module named 'trackio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_trackio)
    track(_record())  # must not raise


def test_replay_sends_every_record(fake_trackio):
    count = replay([_record(), _record(technique="vlad"), _record(technique="fisher")])

    assert count == 3
    assert [call[0] for call in fake_trackio.calls].count("init") == 3
    assert [call[0] for call in fake_trackio.calls].count("finish") == 3


def test_replay_reports_zero_without_trackio(monkeypatch):
    # The whole point of `replay` is rebuilding a store that may not exist on this
    # machine at all; it must report that rather than pretending it worked.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def no_trackio(name, *args, **kwargs):
        if name == "trackio":
            raise ImportError("No module named 'trackio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_trackio)
    assert replay([_record()]) == 0


def test_replay_of_nothing_is_not_an_error(fake_trackio):
    assert replay([]) == 0
    assert fake_trackio.calls == []
