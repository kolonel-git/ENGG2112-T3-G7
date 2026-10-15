"""Produce complete-sensor-row copies of turbine export workbooks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from wt_predmaint.data.turbine_export import (
    NS,
    SENSOR_COLUMNS,
    column_letter,
    publish_workbook,
)

TAG = "{" + NS + "}"
MISSING = {"", "nan", "na", "n/a", "null", "none"}


def value(cell: ET.Element) -> str:
    if cell.get("t") == "inlineStr":
        return "".join(cell.itertext()).strip()
    return (cell.findtext(TAG + "v") or "").strip()


def rows(archive: ZipFile, path: str):
    with archive.open(path) as source:
        for _, element in ET.iterparse(source, events=["end"]):
            if element.tag == TAG + "row":
                yield element
                element.clear()


def clean_workbook(source: Path, output: Path) -> dict:
    """Remove unavailable columns and rows missing any remaining sensor value.

    Requires the export's sidecar report so unavailable features are not guessed.
    Blank alarm fields are valid; alarms are never propagated to other timestamps.
    """
    if source.resolve() == output.resolve():
        raise ValueError("Processed output must differ from the interim workbook")
    metadata = json.loads(source.with_suffix(".report.json").read_text(encoding="utf-8"))
    unavailable = {
        name for names in metadata["missing_features"].values() for name in names
    }
    required = {"StationId", "TimeStamp", *(set(SENSOR_COLUMNS) - unavailable)}
    if not set(SENSOR_COLUMNS) - unavailable:
        raise ValueError("No available sensor features to check")
    report = {"source": str(source.resolve()), "removed_columns": sorted(unavailable),
              "required_columns": sorted(required), "sheets": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        staged = Path(work) / "cleaned.xlsx"
        with ZipFile(source) as original, ZipFile(staged, "w", ZIP_DEFLATED) as result:
            if "xl/sharedStrings.xml" in original.namelist():
                raise ValueError("Use the original turbine exporter workbook (inline strings)")
            workbook = ET.fromstring(original.read("xl/workbook.xml"))
            sheets = workbook.find(TAG + "sheets")
            paths = {f"xl/worksheets/sheet{i}.xml": sheet.get("name")
                     for i, sheet in enumerate(sheets, 1)}
            for entry in original.infolist():
                if entry.filename not in paths:
                    result.writestr(entry, original.read(entry))
                    continue
                iterator = rows(original, entry.filename)
                header = next(iterator)
                columns = {c.get("r").rstrip("0123456789"): value(c) for c in header}
                if required - set(columns.values()):
                    missing_headers = required - set(columns.values())
                    raise ValueError(f"Missing required headers: {missing_headers}")
                retained = [ref for ref, name in columns.items() if name not in unavailable]
                mapping = {ref: column_letter(i) for i, ref in enumerate(retained, 1)}
                required_refs = {ref for ref, name in columns.items() if name in required}
                alarm_refs = {ref for ref, name in columns.items() if name == "AlarmEvents"}
                stats = {"input_rows": 0, "kept_rows": 0, "removed_rows": 0,
                         "alarm_events_in_removed_rows": 0}
                with result.open(entry.filename, "w") as target:
                    target.write((f'<worksheet xmlns="{NS}"><sheetViews>'
                                  '<sheetView workbookViewId="0"><pane ySplit="1" '
                                  'topLeftCell="A2" state="frozen"/></sheetView>'
                                  '</sheetViews><sheetData>').encode())

                    def write_row(
                        row: ET.Element, number: int, mapping: dict = mapping
                    ) -> None:
                        row.set("r", str(number))
                        row.attrib.pop("spans", None)
                        for cell in list(row):
                            ref = cell.get("r").rstrip("0123456789")
                            if ref not in mapping:
                                row.remove(cell)
                            else:
                                cell.set("r", mapping[ref] + str(number))
                        target.write(ET.tostring(row, encoding="utf-8"))

                    write_row(header, 1)
                    for row in iterator:
                        stats["input_rows"] += 1
                        values = {c.get("r").rstrip("0123456789"): value(c) for c in row}
                        if any(values.get(ref, "").lower() in MISSING for ref in required_refs):
                            stats["removed_rows"] += 1
                            for ref in alarm_refs:
                                if values.get(ref):
                                    stats["alarm_events_in_removed_rows"] += len(
                                        json.loads(values[ref])
                                    )
                            continue
                        stats["kept_rows"] += 1
                        write_row(row, stats["kept_rows"] + 1)
                    target.write((f'</sheetData><autoFilter ref="A1:{column_letter(len(retained))}'
                                  f'{stats["kept_rows"] + 1}"/></worksheet>').encode())
                report["sheets"][paths[entry.filename]] = stats
        publish_workbook(staged, output)
    output.with_suffix(".cleaning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
