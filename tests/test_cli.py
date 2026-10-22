import pandas as pd
import pytest
from simdata import make_table

from segmentscan.cli import main, parse_bins


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "table.csv"
    df = make_table(6000, seed=0, planted=0.45)
    df["age"] = range(len(df))
    df.to_csv(path, index=False)
    return path


def test_cli_prints_a_summary(csv_path, capsys):
    code = main([str(csv_path), "--dims", "a,b,c", "--outcome", "y", "--threshold", "0.3"])
    output = capsys.readouterr().out
    assert code == 0
    assert output.startswith("ScanResult:")
    assert "a=a0 & b=b0" in output


def test_cli_writes_all_tested_cells(csv_path, tmp_path, capsys):
    out = tmp_path / "cells.csv"
    code = main(
        [
            str(csv_path),
            "--dims", "a,b,age",
            "--outcome", "y",
            "--threshold", "0.3",
            "--bins", "age:3",
            "--method", "by",
            "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    cells = pd.read_csv(out)
    assert {"cell", "holdout_n", "p_adj", "discovery", "stable"} <= set(cells.columns)
    assert "Wrote" in capsys.readouterr().out


def test_cli_reports_errors_without_a_traceback(csv_path, capsys):
    code = main([str(csv_path), "--dims", "nope", "--outcome", "y", "--threshold", "0.3"])
    assert code == 2
    error = capsys.readouterr().err
    assert error.startswith("segmentscan: error: columns not found in df: ['nope']")


def test_python_dash_m_runs_the_cli(csv_path):
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "segmentscan", str(csv_path), "--dims", "a,b",
         "--outcome", "y", "--threshold", "0.3"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert completed.returncode == 0
    assert completed.stdout.startswith("ScanResult:")


def test_parse_bins():
    assert parse_bins(None) is None
    assert parse_bins("age:4, income:5") == {"age": 4, "income": 5}
    with pytest.raises(ValueError):
        parse_bins("age")
    with pytest.raises(ValueError):
        parse_bins("age:four")
