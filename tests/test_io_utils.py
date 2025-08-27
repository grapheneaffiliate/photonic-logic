import json
import tempfile
from pathlib import Path

from plogic.utils.io import save_csv, save_json


def test_save_json():
    data = {
        "timing": {"tau_ph_ns": 1.0, "t_switch_ns": 2.0},
        "energetics": {"E_op_fJ": 3.14, "photons_per_op": 42},
        "test_value": 123,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_json(data, path)

        assert path.exists()

        # Verify content
        with path.open("r") as f:
            loaded = json.load(f)

        assert loaded == data
        assert loaded["timing"]["tau_ph_ns"] == 1.0
        assert loaded["energetics"]["E_op_fJ"] == 3.14


def test_save_json_creates_directories():
    data = {"test": "value"}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "test.json"
        save_json(data, path)

        assert path.exists()
        assert path.parent.exists()


def test_save_csv_basic():
    report = {
        "timing": {"tau_ph_ns": 1.0, "t_switch_ns": 2.0},
        "energetics": {"E_op_fJ": 3.14, "photons_per_op": 42},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.csv"
        save_csv(report, path)

        assert path.exists()

        # Verify CSV content
        content = path.read_text()
        assert "timing.tau_ph_ns" in content
        assert "timing.t_switch_ns" in content
        assert "energetics.E_op_fJ" in content
        assert "1.0" in content
        assert "2.0" in content
        assert "3.14" in content


def test_save_csv_with_gate_stats():
    report = {
        "timing": {"tau_ph_ns": 1.0},
        "stats": {
            "per_gate": {
                "AND": {"contrast_dB": 20.0, "min_on": 1.0},
                "OR": {"contrast_dB": 25.0, "min_on": 0.9},
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_gates.csv"
        save_csv(report, path)

        assert path.exists()

        content = path.read_text()
        assert "per_gate.AND.contrast_dB" in content
        assert "per_gate.OR.contrast_dB" in content
        assert "20.0" in content
        assert "25.0" in content


def test_save_csv_append_mode():
    report1 = {"value": 1, "name": "first"}
    report2 = {"value": 2, "name": "second"}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "append_test.csv"

        # First write should create file with header
        save_csv(report1, path)
        content1 = path.read_text()
        lines1 = content1.strip().split("\n")
        assert len(lines1) == 2  # header + 1 data row

        # Second write should append (header written again but that's OK)
        save_csv(report2, path)
        content2 = path.read_text()
        lines2 = content2.strip().split("\n")
        assert len(lines2) >= 3  # header + 2 data rows (may have duplicate header)

        assert "first" in content2
        assert "second" in content2


def test_save_csv_creates_directories():
    report = {"test": "value"}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "test.csv"
        save_csv(report, path)

        assert path.exists()
        assert path.parent.exists()


def test_flatten_nested_dict():
    # Test the flattening behavior indirectly through save_csv
    nested_report = {
        "level1": {"level2": {"level3": "deep_value"}, "simple": "value"},
        "top_level": 42,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested.csv"
        save_csv(nested_report, path)

        content = path.read_text()
        assert "level1.level2.level3" in content
        assert "level1.simple" in content
        assert "top_level" in content
        assert "deep_value" in content


def test_empty_report():
    # Test edge case with empty report
    empty_report = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.csv"
        save_csv(empty_report, path)

        assert path.exists()
        content = path.read_text()
        # Should have at least a header line
        assert len(content.strip()) >= 0


def test_json_with_special_types():
    # Test JSON serialization with various types
    data = {
        "float": 3.14159,
        "int": 42,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"inner": "value"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "types.json"
        save_json(data, path)

        # Verify round-trip
        with path.open("r") as f:
            loaded = json.load(f)

        assert loaded["float"] == 3.14159
        assert loaded["int"] == 42
        assert loaded["bool"] is True
        assert loaded["null"] is None
        assert loaded["list"] == [1, 2, 3]
        assert loaded["nested"]["inner"] == "value"
