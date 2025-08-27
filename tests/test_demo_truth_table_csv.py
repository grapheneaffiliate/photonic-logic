from pathlib import Path
from plogic.utils.io import save_truth_table_csv
import csv


def test_save_truth_table_csv(tmp_path: Path):
    """Test saving truth table to CSV with soft and hard outputs."""
    gate = "XOR"
    # simulate a soft XOR with near-ideal analog values
    soft = [3e-7, 0.9999997, 0.9999997, 3e-7]
    hard = [0, 1, 1, 0]
    out = tmp_path / "xor_truth.csv"
    
    save_truth_table_csv(
        gate_name=gate,
        outputs=None,
        logic_out_soft=soft,
        logic_out_hard=hard,
        path=out
    )
    
    # Verify file exists
    assert out.exists()
    
    # Read and verify content
    txt = out.read_text()
    assert "gate,A,B,soft,hard" in txt
    assert "XOR,0,1," in txt
    assert "XOR,1,1," in txt
    
    # Verify CSV structure
    with open(out, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 4  # 4 input combinations
    
    # Check first row (0,0)
    assert rows[0]['gate'] == 'XOR'
    assert rows[0]['A'] == '0'
    assert rows[0]['B'] == '0'
    assert float(rows[0]['soft']) < 1e-6
    assert rows[0]['hard'] == '0'
    
    # Check second row (0,1)
    assert rows[1]['A'] == '0'
    assert rows[1]['B'] == '1'
    assert float(rows[1]['soft']) > 0.999
    assert rows[1]['hard'] == '1'
    
    # Check third row (1,0)
    assert rows[2]['A'] == '1'
    assert rows[2]['B'] == '0'
    assert float(rows[2]['soft']) > 0.999
    assert rows[2]['hard'] == '1'
    
    # Check fourth row (1,1)
    assert rows[3]['A'] == '1'
    assert rows[3]['B'] == '1'
    assert float(rows[3]['soft']) < 1e-6
    assert rows[3]['hard'] == '0'


def test_save_truth_table_csv_outputs_fallback(tmp_path: Path):
    """Test that outputs is used as fallback when logic_out_soft is None."""
    gate = "AND"
    outputs = [0.0, 0.0, 0.0, 1.0]
    hard = [0, 0, 0, 1]
    out = tmp_path / "and_truth.csv"
    
    save_truth_table_csv(
        gate_name=gate,
        outputs=outputs,
        logic_out_soft=None,  # Not provided
        logic_out_hard=hard,
        path=out
    )
    
    # Read and verify content
    with open(out, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 4
    
    # Verify outputs was used for soft column
    assert float(rows[0]['soft']) == 0.0
    assert float(rows[1]['soft']) == 0.0
    assert float(rows[2]['soft']) == 0.0
    assert float(rows[3]['soft']) == 1.0


def test_save_truth_table_csv_creates_directory(tmp_path: Path):
    """Test that parent directories are created if they don't exist."""
    gate = "OR"
    soft = [0.0, 1.0, 1.0, 1.0]
    hard = [0, 1, 1, 1]
    out = tmp_path / "nested" / "dir" / "or_truth.csv"
    
    save_truth_table_csv(
        gate_name=gate,
        outputs=None,
        logic_out_soft=soft,
        logic_out_hard=hard,
        path=out
    )
    
    assert out.exists()
    assert out.parent.exists()
