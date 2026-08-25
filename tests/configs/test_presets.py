"""Presets must name configurations that were actually measured.

A preset is a shorthand for a row in the README, so the thing worth testing is not that
it constructs -- importing the module does that -- but that it still corresponds to a
run in `results/runs.jsonl`. A preset that drifts from the record is worse than no
preset: it looks like the documented configuration and is not.
"""

from __future__ import annotations

import pytest

from cbir.configs.presets import PRESETS
from cbir.configs.run import descriptor_params
from cbir.eval.results import load


@pytest.fixture(scope="module")
def recorded():
    """Every recorded run, as `(technique, params)`."""
    return [(record.technique, record.params) for record in load()]


def _same(value, other) -> bool:
    """JSON round-trips tuples to lists, so compare sequences by content."""
    if isinstance(value, list | tuple) and isinstance(other, list | tuple):
        return list(value) == list(other)
    return value == other


def _matches(config, technique: str, params: dict) -> bool:
    """Does `config` describe the run that recorded `params`?

    Not an equality check, because `params` is a schema snapshot: `weights`,
    `whiten_source`, `shrinkage` and `last_pool` were all added after runs existed, so a
    row from before an option was introduced simply has no key for it. A missing key
    means the run took whatever that field defaults to -- which is why it went
    unrecorded -- so the config must hold the default there for the two to describe the
    same experiment. Any key the row *does* carry has to match outright.
    """
    if config.technique != technique:
        return False
    fields = descriptor_params(config)
    for key, value in fields.items():
        if key in params:
            if not _same(value, params[key]):
                return False
        elif not _same(value, type(config).__dataclass_fields__[key].default):
            return False
    return not set(params) - set(fields)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_matches_a_recorded_run(name, recorded):
    config = PRESETS[name]

    assert any(_matches(config, technique, params) for technique, params in recorded), (
        f"preset {name!r} does not match any row in results/runs.jsonl -- either it drifted "
        f"from the configuration that was measured, or the run was never recorded"
    )


def test_preset_names_are_cli_safe():
    # They become subcommand names; anything tyro would rewrite makes the documented
    # command differ from the one that works.
    for name in PRESETS:
        assert name == name.lower() and " " not in name and "_" not in name, name


def test_the_fine_tuned_presets_carry_the_combination_that_only_works_one_way():
    # The reason presets exist at all: this is ten flags, and getting one wrong is a
    # plausible wrong number rather than an error.
    for name in ("gem-ft-r101", "gem-ft-vgg16"):
        config = PRESETS[name]
        assert config.p == "learned"
        assert config.weights == "sfm120k"
        assert config.whiten and config.whiten_source == "learned"
