import os
import subprocess
import sys
from pathlib import Path

from pm_dfba_sim.figures import PAPER_CONCEPT_FIGURES


def test_paper_figures_command_creates_expected_pngs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pm_dfba_sim.run_paper_figures",
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Wrote 3 paper concept figures" in result.stdout
    for filename in PAPER_CONCEPT_FIGURES:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
