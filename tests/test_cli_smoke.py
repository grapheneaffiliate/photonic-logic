import shutil
import subprocess
import sys
from pathlib import Path


def _run(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_cli_help():
    exe = shutil.which("plogic")
    if exe:
        proc = _run([exe, "--help"])
    else:
        proc = _run([sys.executable, "-m", "plogic.cli", "--help"])
    assert proc.returncode == 0
    assert "Programmable Photonic Logic CLI" in proc.stdout or "usage" in proc.stdout.lower()


def test_truth_table_generates_csv(tmp_path: Path):
    # Minimal, fast run with two control powers
    out = tmp_path / "truth.csv"
    exe = shutil.which("plogic")
    args = ["truth-table", "--ctrl", "0", "0.001", "--out", str(out)]
    if exe:
        proc = _run([exe, *args])
    else:
        proc = _run([sys.executable, "-m", "plogic.cli", *args])

    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert out.exists()
    # Quick sanity on header columns
    head = out.read_text().splitlines()[0]
    assert "P_ctrl_W" in head and "T_through" in head and "T_drop" in head
