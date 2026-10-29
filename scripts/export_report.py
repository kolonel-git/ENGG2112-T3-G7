"""Export a standalone copy of the report: tables inlined, figures flattened into one folder.

``reports/report/main.tex`` reads its tables from ``tables/`` and its figures from sub-folders of
``reports/figures``. This writes ``reports/report/export/main.tex`` (tables pasted in, figure paths
reduced to file names, ``\\graphicspath{{figures/}}``) and copies the PNGs it uses into
``reports/report/export/figures/``, so the folder can be uploaded to a LaTeX editor as it is.
Put ``refs.bib`` and the logo next to the exported ``main.tex``.

    python scripts/export_report.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "report"
FIGURES = ROOT / "reports" / "figures"
OUT = REPORT / "export"
INCLUDE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}")
INPUT = re.compile(r"\\input\{(tables/[^}]+)\}")


def main() -> None:
    text = (REPORT / "main.tex").read_text(encoding="utf-8")
    text = INPUT.sub(lambda m: (REPORT / m.group(1)).read_text(encoding="utf-8").strip(), text)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    def flatten(m: re.Match[str]) -> str:
        name = m.group(2)
        source = FIGURES / name
        if source.exists():
            shutil.copy2(source, OUT / "figures" / source.name)
            copied.append(source.name)
        return f"{m.group(1)}{{{Path(name).name}}}"

    text = INCLUDE.sub(flatten, text)
    text = text.replace(r"\graphicspath{{../figures/}}", r"\graphicspath{{figures/}}")
    (OUT / "main.tex").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'main.tex'} and {len(copied)} figures in {OUT / 'figures'}")


if __name__ == "__main__":
    main()
