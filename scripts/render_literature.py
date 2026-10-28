"""Render the reviewed CSV without creating or inventing literature entries."""

import argparse
import csv
from pathlib import Path


def render(source):
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = [
        "source_id",
        "dimension",
        "representation",
        "construction",
        "assumptions",
        "applications",
        "evaluation",
        "limitations_authors",
        "limitations_our_inference",
        "open_questions",
        "source_url",
        "reading_scope",
    ]
    lines = [
        "# Literature matrix · D1",
        "",
        "Four verified 2D/video sources. D2 adds 3D/4D and manipulation sources;",
        "this is not the final ten-source matrix. CSV is the source of truth.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[key].replace("|", "\\|") for key in columns) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    study = Path(__file__).resolve().parents[1] / "study"
    output = study / "literature_matrix.md"
    rendered = render(study / "literature_matrix.csv")
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("literature_matrix.md is stale; rerun scripts/render_literature.py")
    else:
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
