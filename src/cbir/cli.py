"""cbir CLI entry point."""

from __future__ import annotations

from typing import Literal

import tyro

from cbir.data.revisitop import SUPPORTED, download


def download_cmd(
    datasets: tuple[Literal["roxford5k", "rparis6k", "all"], ...],
) -> None:
    """Download a RevisitOP benchmark split.

    Args:
        datasets: Dataset(s) to download.
    """
    names = list(SUPPORTED) if "all" in datasets else datasets
    for name in names:
        download(name)


def main() -> None:
    tyro.extras.subcommand_cli_from_dict({"download": download_cmd})


if __name__ == "__main__":
    main()
