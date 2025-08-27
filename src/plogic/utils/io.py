from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


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
