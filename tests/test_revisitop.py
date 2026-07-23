import pytest

from cbir.data.revisitop import SUPPORTED, UNSUPPORTED_CONFIGS, download

# These are the verified counts documented in AGENTS.md — a change here must be a
# deliberate, re-verified update to that table, not an incidental edit.
EXPECTED_COUNTS = {
    "roxford5k": (4993, 70),
    "rparis6k": (6322, 70),
}


def test_supported_configs_match_verified_counts():
    assert {name: (spec.expected_db, spec.expected_queries) for name, spec in SUPPORTED.items()} == EXPECTED_COUNTS


@pytest.mark.parametrize("name", sorted(UNSUPPORTED_CONFIGS))
def test_unsupported_configs_are_rejected_without_network(name):
    with pytest.raises(ValueError, match=name):
        download(name)


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        download("not-a-real-dataset")
