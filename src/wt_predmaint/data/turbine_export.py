"""Select SCADA features and export exact-time turbine sheets using only Python's stdlib."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

FEATURES = {
    "tblSCTurPress": [
        "wtc_GearPres_mean", "wtc_HydPress_mean", "wtc_HubPresA_mean",
        "wtc_HubPresB_mean", "wtc_HubPresC_mean",
    ],
    "tblSCTurTemp": [
        "wtc_" + name + "_mean" for name in [
            "HSGenTmp", "HSRotTmp", "IMSGenTm", "IMSRotTm", "GeOilTmp",
            "GenBeGTm", "GenBeRTm", "MainBTmp", "Gen1U1Tm", "Gen1V1Tm",
            "Gen1W1Tm", "AmbieTmp", "NacelTmp", "ConvWTmp", "InvUpTmp",
            "TraOilTF", "TraRooTF", "TrafoTMa", "HydOilTm", "A1ExtTmp",
            "A3LefTmp", "A3RigTmp", "A9IntTmp", "A21IntTm", "DeltaTmp",
            "BrkTmpGn", "GFilB1Tm", "GFilB2Tm", "GFilB3Tm", "ReacUTmp",
            "ReacVTmp", "ReacWTmp",
        ]
    ],
    "tblSCTurbine": [
        "wtc_PrWindSp_mean", "wtc_PrWindSp_stddev", "wtc_GenRpm_mean",
        "wtc_GenRpm_stddev", "wtc_MainSRpm_mean", "wtc_PowerRef_endvalue",
    ],
}
SENSOR_COLUMNS = [column for columns in FEATURES.values() for column in columns]
HEADERS = ["TimeStamp", "StationId", *SENSOR_COLUMNS, "Alarmcode", "AlarmEvents"]
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def publish_workbook(staged: Path, output: Path) -> None:
    """Replace output atomically with a file inheriting the destination folder's ACL.

    Windows Python 3.13 restricts TemporaryDirectory permissions to its creator.
    Moving a file out of that directory retains its restrictive permissions.
    Create a fresh sibling instead, copying bytes before replacing the output.
    """
    sibling = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with sibling.open("xb") as target, staged.open("rb") as source:
            shutil.copyfileobj(source, target)
        sibling.replace(output)
    finally:
        sibling.unlink(missing_ok=True)


def timestamp(value: str) -> str:
    """Validate and normalise timestamps without changing their timezone or rounding."""
    text = value.strip()
    if "/" in text:
        # The supplied October 2018 pressure CSV uses day/month/year.
        date_format = "%d/%m/%Y %H:%M:%S" if text.count(":") == 2 else "%d/%m/%Y %H:%M"
        parsed = datetime.strptime(text, date_format)
    else:
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        raise ValueError("Timezone-aware timestamps require an explicit alignment policy")
    return parsed.isoformat(sep=" ")


def station_id(value: str) -> str:
    value = value.strip()
    if not value or not value.isascii() or not value.isdigit() or len(value) > 23:
        raise ValueError(f"Expected a numeric turbine ID, got {value!r}")
    return value


def column_letter(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell(value: str | None, reference: str, numeric: bool = False) -> str:
    if value is None or value == "":
        return ""
    if len(value) > 32767:
        raise ValueError(f"Excel cell length exceeded at {reference}")
    if numeric:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite sensor value at {reference}: {value}")
        return f'<c r="{reference}"><v>{number!r}</v></c>'
    return (
        f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">'
        f"{escape(value)}</t></is></c>"
    )


def write_workbook(db: sqlite3.Connection, output: Path, stations: list[str]) -> dict:
    counts = {}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        overrides = "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            'spreadsheetml.worksheet+xml"/>' for i in range(1, len(stations) + 1)
        )
        archive.writestr("[Content_Types].xml", (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
            'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.'
            'openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            f"{overrides}</Types>"
        ))
        archive.writestr("_rels/.rels", (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        ))
        sheets = "".join(
            f'<sheet name="Station_{station}" sheetId="{i}" r:id="rId{i}"/>'
            for i, station in enumerate(stations, 1)
        )
        archive.writestr("xl/workbook.xml", (
            f'<workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/'
            f'officeDocument/2006/relationships"><sheets>{sheets}</sheets></workbook>'
        ))
        relationships = "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/'
            f'officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(stations) + 1)
        )
        archive.writestr("xl/_rels/workbook.xml.rels", (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{relationships}</Relationships>"
        ))
        for i, station in enumerate(stations, 1):
            count = db.execute(
                "SELECT COUNT(*) FROM readings WHERE station=?", (station,)
            ).fetchone()[0]
            if count > 1048575:
                raise ValueError(f"Station {station} exceeds Excel's row limit")
            counts[station] = count
            with archive.open(f"xl/worksheets/sheet{i}.xml", "w") as sheet:
                sheet.write((
                    f'<worksheet xmlns="{NS}"><sheetViews><sheetView workbookViewId="0">'
                    '<pane ySplit="1" topLeftCell="A2" state="frozen"/>'
                    '</sheetView></sheetViews><sheetData>'
                ).encode())
                sheet.write(("<row r=\"1\">" + "".join(
                    cell(header, f"{column_letter(j)}1")
                    for j, header in enumerate(HEADERS, 1)
                ) + "</row>").encode())
                query = "SELECT time, payload, alarms FROM readings WHERE station=? ORDER BY time"
                for row_number, (time, payload, alarms) in enumerate(
                    db.execute(query, (station,)), 2
                ):
                    values = json.loads(payload)
                    events = json.loads(alarms)
                    codes = "; ".join(event["Alarmcode"] for event in events)
                    row = [time, station, *(values.get(c) for c in SENSOR_COLUMNS),
                           codes, json.dumps(events) if events else None]
                    xml = "".join(
                        cell(value, f"{column_letter(j)}{row_number}",
                             numeric=3 <= j <= 2 + len(SENSOR_COLUMNS))
                        for j, value in enumerate(row, 1)
                    )
                    sheet.write(f'<row r="{row_number}">{xml}</row>'.encode())
                last_column = column_letter(len(HEADERS))
                sheet.write((f'</sheetData><autoFilter ref="A1:{last_column}{count + 1}"/>'
                             '</worksheet>').encode())
    return counts


def export_turbines(input_dir: Path, output: Path, pattern: str = "*.csv") -> dict:
    """Find CSVs recursively and outer-join sensors and alarms on exact time keys."""
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"inputs": {}, "missing_features": {}, "exact_duplicates_removed": {},
              "rounded_duplicates_merged": {},
              "alarm_alignment": "Exact TimeOn; StationNr mapped to StationId"}
    with tempfile.TemporaryDirectory(prefix="turbine_export_", dir=output.parent) as work:
        db = sqlite3.connect(str(Path(work) / "readings.sqlite"))
        try:
            db.execute("CREATE TABLE readings (station TEXT, time TEXT, payload TEXT, "
                       "alarms TEXT, PRIMARY KEY(station,time))")
            db.execute("CREATE TABLE seen (table_name TEXT, station TEXT, time TEXT, "
                       "payload TEXT, PRIMARY KEY(table_name,station,time))")
            sensor_stations: set[str] = set()
            alarm_stations: set[str] = set()
            for table in [*FEATURES, "tblAlarmLog"]:
                paths = sorted(p for p in input_dir.rglob(pattern)
                               if p.is_file() and p.name.startswith(table + "_"))
                if not paths:
                    if table not in FEATURES:
                        raise ValueError(f"No input files for {table} matching {pattern}")
                    print(f"No {table} files found; its sensor features are unavailable",
                          flush=True)
                    report["inputs"][table] = []
                    report["missing_features"][table] = list(FEATURES[table])
                    report["exact_duplicates_removed"][table] = 0
                    report["rounded_duplicates_merged"][table] = 0
                    continue
                report["inputs"][table] = [str(p.resolve()) for p in paths]
                missing: set[str] = set()
                duplicates = 0
                rounded_duplicates = 0
                for path in paths:
                    print(f"Reading {path.name}", flush=True)
                    with path.open(encoding="utf-8-sig", newline="") as source:
                        reader = csv.DictReader(source)
                        fields = set(reader.fieldnames or [])
                        alarm = table == "tblAlarmLog"
                        required = {"StationNr", "TimeOn", "TimeOff", "Alarmcode"} if alarm else {
                            "StationId", "TimeStamp"
                        }
                        if required - fields:
                            raise ValueError(f"{path.name}: missing keys {required - fields}")
                        if not alarm:
                            missing.update(set(FEATURES[table]) - fields)
                        for line, row in enumerate(reader, 2):
                            try:
                                station = station_id(row["StationNr" if alarm else "StationId"])
                                time = timestamp(row["TimeOn" if alarm else "TimeStamp"])
                                existing = db.execute(
                                    "SELECT payload,alarms FROM readings "
                                    "WHERE station=? AND time=?",
                                    (station, time),
                                ).fetchone()
                                if existing:
                                    values, events = (json.loads(v) for v in existing)
                                else:
                                    values, events = {}, []
                                if alarm:
                                    alarm_stations.add(station)
                                    code = row["Alarmcode"].strip()
                                    if not code:
                                        raise ValueError("Empty Alarmcode")
                                    events.append({"Alarmcode": code, "TimeOn": time,
                                                   "TimeOff": timestamp(row["TimeOff"])
                                                   if row["TimeOff"].strip() else None})
                                else:
                                    sensor_stations.add(station)
                                    selected = {c: row[c].strip() for c in FEATURES[table]
                                                if c in fields}
                                    for value in selected.values():
                                        if value and not math.isfinite(float(value)):
                                            raise ValueError(f"Non-finite sensor value: {value}")
                                    serialised = json.dumps(selected, sort_keys=True)
                                    previous = db.execute(
                                        "SELECT payload FROM seen WHERE table_name=? "
                                        "AND station=? AND time=?", (table, station, time),
                                    ).fetchone()
                                    if previous:
                                        if previous[0] != serialised:
                                            old = json.loads(previous[0])
                                            equivalent = old.keys() == selected.keys() and all(
                                                old[c] == selected[c] or (
                                                    old[c] and selected[c] and math.isclose(
                                                        float(old[c]), float(selected[c]),
                                                        rel_tol=1e-9, abs_tol=1e-8,
                                                    )
                                                ) for c in selected
                                            )
                                            if not equivalent:
                                                raise ValueError("Conflicting duplicate sensor key")
                                            selected = {c: max((old[c], selected[c]), key=len)
                                                        for c in selected}
                                            serialised = json.dumps(selected, sort_keys=True)
                                            rounded_duplicates += 1
                                            db.execute(
                                                "UPDATE seen SET payload=? WHERE table_name=? "
                                                "AND station=? AND time=?",
                                                (serialised, table, station, time),
                                            )
                                        else:
                                            duplicates += 1
                                            continue
                                    else:
                                        db.execute("INSERT INTO seen VALUES (?,?,?,?)",
                                                   (table, station, time, serialised))
                                    values.update(selected)
                                db.execute("INSERT OR REPLACE INTO readings VALUES (?,?,?,?)",
                                           (station, time, json.dumps(values), json.dumps(events)))
                            except (ValueError, TypeError, AttributeError) as exc:
                                raise ValueError(f"{path.name}, line {line}: {exc}") from exc
                    db.commit()
                report["missing_features"][table] = sorted(missing)
                report["exact_duplicates_removed"][table] = duplicates
                report["rounded_duplicates_merged"][table] = rounded_duplicates
            report["alarm_only_station_ids"] = sorted(alarm_stations - sensor_stations)
            stations = [r[0] for r in db.execute("SELECT DISTINCT station FROM readings")]
            stations.sort(key=int)
            if not stations:
                raise ValueError("Input files contain no observations")
            # Build completely before replacing an existing output.
            staged = Path(work) / "turbines.xlsx"
            report["rows_per_station"] = write_workbook(db, staged, stations)
            report["alarm_only_rows"] = db.execute(
                "SELECT COUNT(*) FROM readings WHERE payload='{}'"
            ).fetchone()[0]
            publish_workbook(staged, output)
            output.with_suffix(".report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        finally:
            db.close()
    return report
