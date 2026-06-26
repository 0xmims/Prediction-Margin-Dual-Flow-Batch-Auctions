from __future__ import annotations

import argparse
from pathlib import Path

from pm_dfba_sim.figures import generate_paper_concept_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static explanatory concept figures for the PM-DFBA paper draft."
    )
    parser.add_argument(
        "--out",
        default="outputs/paper_figures",
        help="Directory where PNG concept figures should be written.",
    )
    args = parser.parse_args()

    paths = generate_paper_concept_figures(Path(args.out))
    print(f"Wrote {len(paths)} paper concept figures to {Path(args.out)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
