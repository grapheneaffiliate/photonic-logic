import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _plogic_cmd():
    exe = shutil.which("plogic")
    if exe:
        return [exe]
    return [sys.executable, "-m", "plogic.cli"]


def test_truth_table_respects_out_and_writes_csv(tmp_path: Path):
    out = tmp_path / "truth.csv"
    cmd = _plogic_cmd() + ["truth-table", "--ctrl", "0", "0.001", "--out", str(out)]
    proc = _run(cmd)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert out.exists(), "CSV was not created"
    header = out.read_text().splitlines()[0]
    assert "P_ctrl_W" in header and "T_through" in header and "T_drop" in header


def test_characterize_writes_json_report(tmp_path: Path):
    report = tmp_path / "report.json"
    cmd = _plogic_cmd() + ["characterize", "--stages", "1", "--report", str(report)]
    proc = _run(cmd)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert report.exists(), "Report JSON was not created"
    data = json.loads(report.read_text())
    assert "device_parameters" in data and "performance_metrics" in data
