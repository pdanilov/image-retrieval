import numpy as np
import pytest

import cbir.eval.results as results_module
import cbir.eval.runner as runner_module
from cbir.configs.classic import BoWConfig, RunConfig, VLADConfig
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs
from cbir.eval.results import load
from cbir.eval.runner import ground_truth, run, run_all


class _FakeQueryDataset:
    """Just the three ground-truth columns `ground_truth` reads."""

    def __init__(self, easy, hard, junk):
        self._columns = {"easy": easy, "hard": hard, "junk": junk}

    def __getitem__(self, column):
        return self._columns[column]


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Replace everything expensive: SIFT extraction, the dataset load, and the record path.

    The runner's job is composition -- that the right arrays reach the right call in
    the right order -- so every piece it composes is faked here and asserted on. Real
    extraction over ~5k images takes tens of minutes and has its own tests.
    """
    monkeypatch.setattr(results_module, "RESULTS_PATH", tmp_path / "runs.jsonl")

    inputs = ClassicDescriptorInputs(
        held_out_dataset="rparis6k",
        held_out_descriptors=np.zeros((10, 2), dtype=np.float32),
        database_descriptors=[np.zeros((1, 2), dtype=np.float32)] * 3,
        query_descriptors=[np.zeros((1, 2), dtype=np.float32)] * 2,
    )
    # Patched on the config rather than the runner: preparation now belongs to the
    # descriptor, which is what lets a CNN config ask for images instead of RootSIFT.
    # Patching here exercises that dispatch instead of bypassing it.
    prepared = []

    def fake_prepare(self, dataset):
        prepared.append(dataset)
        return inputs

    for config_type in (BoWConfig, VLADConfig):
        monkeypatch.setattr(config_type, "prepare", fake_prepare, raising=True)

    # Query 0 ranks the database 2 > 1 > 0, query 1 ranks it 0 > 1 > 2. The weights are
    # deliberately distinct rather than one-hot: with one-hot queries two of the three
    # similarities tie at 0.0 and the ranking below the top hit depends on argsort's
    # tie-breaking, which would make the expected mAPs unstable.
    database_vectors = np.eye(3, dtype=np.float32)
    query_vectors = np.array([[0.1, 0.2, 0.9], [0.9, 0.2, 0.1]], dtype=np.float32)
    monkeypatch.setattr(
        BoWConfig, "train_and_encode", lambda self, inputs: (database_vectors, query_vectors), raising=True
    )

    # Easy positives rank first for both queries; the hard ones rank last, so the three
    # protocols score differently and a run that mixed them up would show it.
    query_dataset = _FakeQueryDataset(easy=[[2], [0]], hard=[[0], [2]], junk=[[], []])
    monkeypatch.setattr(runner_module, "download", lambda name: (query_dataset, None))

    tracked = []
    monkeypatch.setattr(runner_module, "track", tracked.append)
    return {"prepared": prepared, "tracked": tracked, "query_dataset": query_dataset, "inputs": inputs}


def test_ground_truth_keeps_query_order():
    dataset = _FakeQueryDataset(easy=[[1, 2], [3]], hard=[[4], []], junk=[[], [5]])
    truth = ground_truth(dataset)

    assert [list(gt.easy) for gt in truth] == [[1, 2], [3]]
    assert [list(gt.hard) for gt in truth] == [[4], []]
    assert [list(gt.junk) for gt in truth] == [[], [5]]


def test_run_scores_all_three_protocols(wired):
    record = run(RunConfig(descriptor=BoWConfig(k=3, seed=0)))

    assert set(record.metrics) == {"easy", "medium", "hard"}
    # Hand-computed for the fixture's rankings ([2,1,0] and [0,1,2]), under the
    # benchmarks' trapezoidal AP (see compute_ap -- not step-function AP):
    #   Easy   positive={easy}, ignored={hard}: 0-based ranks [0]    -> AP 1.0
    #   Hard   positive={hard}, ignored={easy}: 0-based ranks [1]    -> (0 + 1/2)/2 = 0.25
    #   Medium both positive, nothing ignored:  0-based ranks [0, 2] ->
    #          (1+1)/2*0.5 + (1/2 + 2/3)/2*0.5 = 0.791666...
    # Three protocols reading the same number would mean they were never applied.
    assert record.metrics["easy"]["map"] == pytest.approx(1.0)
    assert record.metrics["medium"]["map"] == pytest.approx(0.5 + (0.5 + 2 / 3) / 4)
    assert record.metrics["hard"]["map"] == pytest.approx(0.25)
    assert record.metrics["easy"]["num_queries"] == 2


def test_a_protocol_with_no_positives_excludes_rather_than_scores_zero(wired, monkeypatch):
    no_hard = _FakeQueryDataset(easy=[[2], [0]], hard=[[], []], junk=[[], []])
    monkeypatch.setattr(runner_module, "download", lambda name: (no_hard, None))

    record = run(RunConfig(descriptor=BoWConfig(k=3)))

    # Scoring an unanswerable query as 0.0 would drag the mean down and make Hard look
    # worse than it is; the queries are dropped from both averages instead.
    assert record.metrics["hard"]["num_excluded"] == 2
    assert record.metrics["hard"]["num_queries"] == 0
    assert record.metrics["easy"]["num_excluded"] == 0


def test_run_records_the_derived_held_out_dataset_not_the_eval_one(wired):
    record = run(RunConfig(descriptor=BoWConfig(k=3), dataset="roxford5k"))

    assert record.dataset == "roxford5k"
    assert record.held_out_dataset == "rparis6k"
    assert wired["prepared"] == ["roxford5k"]


def test_run_records_technique_and_params_from_the_config(wired):
    record = run(RunConfig(descriptor=BoWConfig(k=1234, seed=7)))

    assert record.technique == "bow"
    assert record.params == {"k": 1234, "seed": 7}
    assert record.seconds is not None


def test_run_appends_exactly_one_row(wired):
    run(RunConfig(descriptor=BoWConfig(k=3)))
    assert len(load()) == 1


def test_run_without_record_writes_nothing_and_tracks_nothing(wired):
    record = run(RunConfig(descriptor=BoWConfig(k=3)), record=False)

    assert load() == []
    assert wired["tracked"] == []
    # The record is still returned -- a scratch run you can inspect, just not one that
    # enters the committed history.
    assert record.metrics["easy"]["map"] == pytest.approx(1.0)


def test_run_mirrors_the_recorded_row_to_tracking(wired):
    record = run(RunConfig(descriptor=BoWConfig(k=3)))
    assert wired["tracked"] == [record]


def test_mp_at_k_reaches_the_metrics(wired):
    record = run(RunConfig(descriptor=BoWConfig(k=3), mp_at_k=5))
    assert record.metrics["easy"]["k"] == 5


def test_run_all_records_every_config_in_order(wired):
    configs = [RunConfig(descriptor=BoWConfig(k=k)) for k in (3, 5, 7)]

    records = run_all(configs)

    assert [record.params["k"] for record in records] == [3, 5, 7]
    # Each finished run is on disk before the next starts, which is what makes a sweep
    # that dies halfway keep its survivors.
    assert [record.params["k"] for record in load()] == [3, 5, 7]


def test_run_all_extracts_once_per_dataset_not_once_per_config(wired):
    # The cached descriptor blobs are ~20 GB for one eval direction, so re-preparing
    # per configuration re-reads all of it for every point of a sweep. This test
    # previously asserted one call *per config* -- it pinned the waste rather than
    # catching it.
    run_all([RunConfig(descriptor=BoWConfig(k=k)) for k in (3, 5, 7)])
    assert wired["prepared"] == ["roxford5k"]


def test_run_all_still_extracts_once_for_each_distinct_dataset(wired):
    # Sharing is keyed by dataset: roxford5k's descriptors must never be handed to a
    # run evaluating rparis6k, which would score one dataset's queries against the
    # other's database.
    run_all(
        [
            RunConfig(descriptor=BoWConfig(k=3), dataset="roxford5k"),
            RunConfig(descriptor=BoWConfig(k=3), dataset="rparis6k"),
            RunConfig(descriptor=BoWConfig(k=5), dataset="roxford5k"),
        ]
    )
    assert wired["prepared"] == ["roxford5k", "rparis6k"]


def test_run_uses_supplied_inputs_without_re_extracting(wired):
    run(RunConfig(descriptor=BoWConfig(k=3)), inputs=wired["inputs"])
    assert wired["prepared"] == []


def test_search_receives_queries_first(monkeypatch, wired):
    # exact_search(queries, database) -- swapping the two still returns a full ranking
    # matrix, just a transposed and meaningless one, and with a square fake it would
    # not even raise. Pin the shapes: 2 queries, 3 database images.
    seen = {}

    def fake_search(queries, database, device=None):
        seen["queries"], seen["database"] = queries.shape, database.shape
        # A full permutation per query, as exact_search returns -- a ranking that omits
        # database indices would leave positives unfindable and crash scoring.
        return np.tile(np.arange(len(database), dtype=np.int64), (len(queries), 1))

    monkeypatch.setattr(runner_module, "exact_search", fake_search)
    run(RunConfig(descriptor=BoWConfig(k=3)))

    assert seen["queries"] == (2, 3)
    assert seen["database"] == (3, 3)


def test_vlad_config_flows_through_unchanged(monkeypatch, wired):
    monkeypatch.setattr(
        VLADConfig,
        "train_and_encode",
        lambda self, inputs: (np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)[:2]),
    )

    record = run(RunConfig(descriptor=VLADConfig(k=64, seed=1, intra_norm=False, power=0.5)))

    assert record.technique == "vlad"
    assert record.params == {"k": 64, "seed": 1, "intra_norm": False, "power": 0.5}


def test_run_all_prepares_separately_per_inputs_kind(wired, monkeypatch):
    # The classic tier wants RootSIFT arrays; the CNN tier wants images. Sharing keyed
    # on dataset alone would hand one tier the other's inputs -- silently, since both
    # arrive as an opaque `inputs` argument.
    other_prepared = []

    class _ImageConfig(BoWConfig):
        technique = "stub-cnn"
        inputs_kind = "images"

        def prepare(self, dataset):
            other_prepared.append(dataset)
            return "images-for-" + dataset

        def train_and_encode(self, inputs):
            assert inputs == "images-for-roxford5k"  # never the classic descriptors
            return np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)[:2]

    run_all(
        [
            RunConfig(descriptor=BoWConfig(k=3), dataset="roxford5k"),
            RunConfig(descriptor=_ImageConfig(k=3), dataset="roxford5k"),
            RunConfig(descriptor=BoWConfig(k=5), dataset="roxford5k"),
        ]
    )

    # One classic preparation shared by both BoW runs, one image preparation for the
    # other tier -- same dataset, two kinds, no cross-contamination.
    assert wired["prepared"] == ["roxford5k"]
    assert other_prepared == ["roxford5k"]
