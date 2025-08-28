from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from _flatten_dict(v, prefix=key)
        else:
            yield key, v


def _flatten_with_gate_stats(report: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(_flatten_dict(report))
    # If controller attached per-gate stats under report["stats"]["per_gate"], expand them
    try:
        per_gate = report["stats"]["per_gate"]
        if isinstance(per_gate, dict):
            for gname, gstats in per_gate.items():
                for k, v in gstats.items():
                    flat[f"per_gate.{gname}.{k}"] = v
    except Exception:
        pass
    return flat


def save_csv(report: Dict[str, Any], path: str | Path) -> None:
    flat = _flatten_with_gate_stats(report)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def save_truth_table_csv(
    gate_name: str,
    outputs: List[float] | None,
    logic_out_soft: List[float] | None,
    logic_out_hard: List[int] | None,
    path: str | Path,
) -> None:
    """
    Save a single gate truth table to CSV with inputs (A,B) rows: 00,01,10,11.
    Columns:
      gate, A, B, soft, hard
    soft = analog output in [0,1] if available
    hard = 0/1 if available
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Prefer explicit 'logic_out_soft' if present; otherwise fall back to normalized 'outputs'
    soft_vals = None
    if logic_out_soft is not None:
        soft_vals = logic_out_soft
    elif outputs is not None:
        soft_vals = outputs

    # Construct rows (4 canonical input patterns)
    patterns = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = []
    for idx, (a, b) in enumerate(patterns):
        row = {
            "gate": gate_name,
            "A": a,
            "B": b,
            "soft": (None if soft_vals is None else soft_vals[idx]),
            "hard": (None if logic_out_hard is None else logic_out_hard[idx]),
        }
        rows.append(row)

    # Write CSV (always with header, overwrite)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gate", "A", "B", "soft", "hard"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
