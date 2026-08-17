import json

import pytest

import cbir.eval.results as results_module
from cbir.eval.metrics import EvalResult
from cbir.eval.results import RunRecord, append, format_param, iter_rows, latest, load


@pytest.fixture
def runs_path(tmp_path, monkeypatch):
    """Point the results store at a temp file, never the real results/runs.jsonl."""
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(results_module, "RESULTS_PATH", path)
    return path


def _record(k: int = 5000, technique: str = "bow", dataset: str = "roxford5k", **kwargs) -> RunRecord:
    return RunRecord(
        dataset=dataset,
        held_out_dataset="rparis6k",
        technique=technique,
        params={"k": k, "seed": 0},
        metrics={"medium": {"map": 0.1484, "mp_at_k": 0.305, "k": 10, "num_queries": 70, "num_excluded": 0}},
        **kwargs,
    )


def test_append_then_load_round_trip(runs_path):
    record = _record()
    append(record, runs_path)

    loaded = load(runs_path)
    assert len(loaded) == 1
    assert loaded[0] == record


def test_load_is_empty_when_nothing_recorded_yet(runs_path):
    assert load(runs_path) == []


def test_append_creates_the_results_directory(tmp_path):
    nested = tmp_path / "results" / "runs.jsonl"
    append(_record(), nested)
    assert nested.exists()


def test_append_never_rewrites_earlier_rows(runs_path):
    # The append-only property is what makes git diffs conflict-free and history
    # readable, so it is pinned rather than assumed.
    append(_record(k=5000), runs_path)
    first_line = runs_path.read_text().splitlines()[0]

    append(_record(k=20000), runs_path)
    lines = runs_path.read_text().splitlines()

    assert len(lines) == 2
    assert lines[0] == first_line


def test_one_json_object_per_line(runs_path):
    append(_record(k=5000), runs_path)
    append(_record(k=20000), runs_path)

    rows = [json.loads(line) for line in runs_path.read_text().splitlines()]
    assert [row["params"]["k"] for row in rows] == [5000, 20000]


def test_latest_returns_the_most_recent_row_per_configuration(runs_path):
    # Re-running a configuration appends rather than replaces; `latest` is what
    # resolves "what does this score now?" without discarding the history.
    append(_record(k=5000, recorded_at="2026-01-01T00:00:00+00:00"), runs_path)
    append(_record(k=20000), runs_path)
    rerun = _record(k=5000, recorded_at="2026-06-01T00:00:00+00:00")
    append(rerun, runs_path)

    current = latest(runs_path)
    assert len(current) == 2
    assert current[rerun.key] == rerun
    assert current[rerun.key].recorded_at == "2026-06-01T00:00:00+00:00"


def test_key_distinguishes_dataset_technique_and_params():
    base = _record(k=5000)
    assert base.key == _record(k=5000).key
    assert base.key != _record(k=20000).key
    assert base.key != _record(technique="vlad").key
    assert base.key != _record(dataset="rparis6k").key


def test_key_ignores_param_ordering():
    a = RunRecord("d", "h", "bow", {"k": 64, "seed": 0}, {})
    b = RunRecord("d", "h", "bow", {"seed": 0, "k": 64}, {})
    assert a.key == b.key


def test_from_eval_results_keys_metrics_by_protocol():
    evals = [
        EvalResult(protocol="easy", k=10, map=0.1775, mp_at_k=0.3211, num_queries=68, num_excluded=2),
        EvalResult(protocol="medium", k=10, map=0.1484, mp_at_k=0.3050, num_queries=70, num_excluded=0),
        EvalResult(protocol="hard", k=10, map=0.0570, mp_at_k=0.0855, num_queries=70, num_excluded=0),
    ]
    record = RunRecord.from_eval_results(
        dataset="roxford5k",
        held_out_dataset="rparis6k",
        technique="bow",
        params={"k": 5000, "seed": 0},
        results=evals,
        seconds=2344.6,
    )

    assert set(record.metrics) == {"easy", "medium", "hard"}
    assert record.metrics["easy"]["map"] == pytest.approx(0.1775)
    # num_excluded travels with the score: a protocol that drops queries would
    # otherwise look like a better number over a smaller set.
    assert record.metrics["easy"]["num_excluded"] == 2
    assert record.metrics["hard"]["num_queries"] == 70
    assert record.seconds == pytest.approx(2344.6)


def test_iter_rows_tolerates_fields_this_version_does_not_know(runs_path):
    # Rows written by a future schema must stay readable for analysis; load() would
    # raise on the unknown field, iter_rows() must not.
    append(_record(), runs_path)
    with runs_path.open("a") as f:
        f.write(json.dumps({"dataset": "roxford5k", "some_future_field": 1}) + "\n")

    rows = list(iter_rows(runs_path))
    assert len(rows) == 2
    assert rows[1]["some_future_field"] == 1

    with pytest.raises(TypeError):
        load(runs_path)


def test_format_param_leaves_scalars_alone():
    assert format_param(5000) == "5000"
    assert format_param(None) == "None"
    assert format_param("alexnet") == "alexnet"


def test_format_param_flattens_a_sequence_without_spaces():
    # The results table separates params with spaces and the trackio run name with
    # dashes, so a tuple's default repr would split one param across several columns.
    assert format_param((1.0, 2**-0.5, 0.5)) == "1|0.707107|0.5"


def test_format_param_renders_a_json_round_trip_identically():
    # json turns a tuple into a list; if the two rendered differently, a reloaded row
    # would not visibly match the config that produced it.
    scales = (1.0, 2**-0.5, 0.5)
    assert format_param(json.loads(json.dumps(scales))) == format_param(scales)
