"""Render the reviewed CSV without creating or inventing literature entries."""

import argparse
import csv
from pathlib import Path


def render(source):
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = [
        "source_id",
        "title",
        "year",
        "venue",
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
        "# Part 1a · Literature matrix",
        "",
        f"{len(rows)} source-verified entries spanning 2D/video, 3D mapping, manipulation and graph/VLA interfaces.",
        "CSV is the source of truth. Verification and reading scopes vary by row; abstract-only",
        "inspection does not establish full-paper review or the student's personal reading progress.",
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
