"""Generates the configargparse config file that
molpal.objectives.lookup.LookupObjective expects, pointing it at this
target's cached DOCKSTRING pool (src/data/dockstring.py builds/loads the same
file). The cache's column layout (title line, smiles col 0, score col 1)
matches LookupObjective's defaults, so the config only needs `--path`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.data.dockstring import cache_path, load_oracle

ROOT = Path(__file__).resolve().parent.parent.parent


def write_objective_config(target: str, out_path: Optional[Path] = None) -> str:
    """Ensure the target's pool cache exists, then write a LookupObjective
    config file pointing at it. Returns the config filepath.
    """
    load_oracle(target)  # builds data/dockstring/{target}.csv.gz if missing

    out_path = out_path or (ROOT / "data" / "dockstring" / f"{target}_objective.ini")
    out_path.write_text(f"path = {cache_path(target)}\n")
    return str(out_path)
