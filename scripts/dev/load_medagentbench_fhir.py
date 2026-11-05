"""Load and verify the MedAgentBench FHIR tables in the isolated fhir schema."""

import csv
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

try:
    from lumilake_server.workloads.medagentbench import translate_search
except ModuleNotFoundError:
    from medagentbench import (
        translate_search,  # type: ignore[import-not-found,no-redef]
    )

FHIR_BASE = "http://10.10.40.1:8080/fhir"
OUTPUT_DIR = Path("/data/lumilake-experiment/medagentbench")
TASK_FILE = OUTPUT_DIR / "test_data_v2.json"
ENV_FILE = Path("/data/lumid-data-app/.env")
WORK_DIR = OUTPUT_DIR / "work"
LOG_PATH = OUTPUT_DIR / "load.log"
RECORD_PATH = OUTPUT_DIR / "run-record.md"
PAGE_SIZE = 1000
SGT = timezone(timedelta(hours=8), name="SGT")

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS fhir;
CREATE TABLE IF NOT EXISTS fhir.patient (id text PRIMARY KEY, identifier text,
 name_family text, name_given text, birth_date text, gender text);
CREATE TABLE IF NOT EXISTS fhir.observation (id text PRIMARY KEY,
 patient_id text NOT NULL, code text, code_text text, category text,
 effective_date_time text, value_quantity_value double precision,
 value_quantity_unit text, value_string text, status text);
CREATE TABLE IF NOT EXISTS fhir.condition (id text PRIMARY KEY,
 patient_id text NOT NULL, code_text text, clinical_status text,
 category text, recorded_date text);
CREATE TABLE IF NOT EXISTS fhir.medication_request (id text PRIMARY KEY,
 patient_id text NOT NULL, medication_text text, medication_code text,
 authored_on text, status text, intent text, dosage_text text);
CREATE TABLE IF NOT EXISTS fhir.procedure (id text PRIMARY KEY,
 patient_id text NOT NULL, code text, code_text text,
 performed_date_time text, status text);
CREATE TABLE IF NOT EXISTS fhir.service_request (id text PRIMARY KEY,
 patient_id text NOT NULL, code text, code_text text, authored_on text,
 status text, intent text, priority text, note_text text,
 occurrence_date_time text);
CREATE INDEX IF NOT EXISTS patient_patient_id_idx ON fhir.patient (id);
CREATE INDEX IF NOT EXISTS patient_date_idx ON fhir.patient (birth_date);
CREATE INDEX IF NOT EXISTS observation_patient_id_idx ON fhir.observation (patient_id);
CREATE INDEX IF NOT EXISTS observation_date_idx
 ON fhir.observation (effective_date_time);
CREATE INDEX IF NOT EXISTS condition_patient_id_idx ON fhir.condition (patient_id);
CREATE INDEX IF NOT EXISTS condition_date_idx ON fhir.condition (recorded_date);
CREATE INDEX IF NOT EXISTS medication_request_patient_id_idx
 ON fhir.medication_request (patient_id);
CREATE INDEX IF NOT EXISTS medication_request_date_idx
 ON fhir.medication_request (authored_on);
CREATE INDEX IF NOT EXISTS procedure_patient_id_idx ON fhir.procedure (patient_id);
CREATE INDEX IF NOT EXISTS procedure_date_idx ON fhir.procedure (performed_date_time);
CREATE INDEX IF NOT EXISTS service_request_patient_id_idx
 ON fhir.service_request (patient_id);
CREATE INDEX IF NOT EXISTS service_request_date_idx
 ON fhir.service_request (authored_on);
"""

TABLE_COLUMNS = {
    "patient": [
        "id",
        "identifier",
        "name_family",
        "name_given",
        "birth_date",
        "gender",
    ],
    "observation": [
        "id",
        "patient_id",
        "code",
        "code_text",
        "category",
        "effective_date_time",
        "value_quantity_value",
        "value_quantity_unit",
        "value_string",
        "status",
    ],
    "condition": [
        "id",
        "patient_id",
        "code_text",
        "clinical_status",
        "category",
        "recorded_date",
    ],
    "medication_request": [
        "id",
        "patient_id",
        "medication_text",
        "medication_code",
        "authored_on",
        "status",
        "intent",
        "dosage_text",
    ],
    "procedure": [
        "id",
        "patient_id",
        "code",
        "code_text",
        "performed_date_time",
        "status",
    ],
    "service_request": [
        "id",
        "patient_id",
        "code",
        "code_text",
        "authored_on",
        "status",
        "intent",
        "priority",
        "note_text",
        "occurrence_date_time",
    ],
}


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: str
    name: str
    user: str
    password: str


def log(message: str, *, also_print: bool = True) -> None:
    line = f"{datetime.now(SGT).isoformat(timespec='seconds')} {message}"
    with LOG_PATH.open("a") as handle:
        handle.write(line + "\n")
    if also_print:
        print(line, flush=True)


def read_db_config() -> DatabaseConfig:
    values: dict[str, str] = {}
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    password = values.get("DB_PASSWORD")
    if not password:
        raise RuntimeError(f"DB_PASSWORD is missing from {ENV_FILE}")
    return DatabaseConfig("127.0.0.1", "5102", "lumilake", "postgres", password)


def psql_env(db: DatabaseConfig) -> dict[str, str]:
    return {**os.environ, "PGPASSWORD": db.password}


def psql(db: DatabaseConfig, sql: str) -> str:
    result = subprocess.run(
        [
            "psql",
            "-h",
            db.host,
            "-p",
            db.port,
            "-U",
            db.user,
            "-d",
            db.name,
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
            "-c",
            sql,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=psql_env(db),
    )
    return result.stdout


def load_mrns() -> list[str]:
    tasks = json.loads(TASK_FILE.read_text())
    mrns = sorted(
        {str(task["eval_MRN"]) for task in tasks if task.get("eval_MRN") is not None}
    )
    if len(mrns) != 98:
        raise RuntimeError(f"expected 98 distinct eval_MRN values, found {len(mrns)}")
    return mrns


def fetch_bundle(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    bundle = response.json()
    if not isinstance(bundle, dict):
        raise RuntimeError(f"FHIR response is not an object: {url}")
    return bundle


def next_link(bundle: Mapping[str, Any]) -> str:
    for link in bundle.get("link", []):
        if isinstance(link, dict) and link.get("relation") == "next":
            value = link.get("url")
            if isinstance(value, str):
                return value.replace("http://localhost:8080", "http://10.10.40.1:8080")
    return ""


def iter_resources(resource: str, mrns: list[str]) -> Iterator[dict[str, Any]]:
    for index, mrn in enumerate(mrns, start=1):
        params = (
            {"identifier": mrn, "_count": PAGE_SIZE, "_format": "json"}
            if resource == "Patient"
            else {"patient": mrn, "_count": PAGE_SIZE, "_format": "json"}
        )
        url = f"{FHIR_BASE}/{resource}?" + urlencode(params)
        while url:
            bundle = fetch_bundle(url)
            for entry in bundle.get("entry", []):
                row = entry.get("resource")
                if isinstance(row, dict):
                    yield row
            url = next_link(bundle)
        if index % 10 == 0 or index == len(mrns):
            log(f"{resource}: fetched {index}/{len(mrns)} MRNs")


def first_coding(concept: Any) -> Mapping[str, Any]:
    if not isinstance(concept, dict):
        return {}
    coding = concept.get("coding")
    if isinstance(coding, list):
        for item in coding:
            if isinstance(item, dict):
                return item
    return {}


def subject_id(resource: Mapping[str, Any]) -> str | None:
    subject = resource.get("subject")
    ref = subject.get("reference") if isinstance(subject, dict) else None
    return ref.removeprefix("Patient/") if isinstance(ref, str) else None


def category_code(resource: Mapping[str, Any]) -> str:
    categories = resource.get("category")
    if isinstance(categories, list):
        for category in categories:
            code = first_coding(category).get("code")
            if isinstance(code, str):
                return code
    return ""


def flatten_patient(row: Mapping[str, Any]) -> dict[str, Any]:
    identifier = ""
    ids = row.get("identifier")
    if isinstance(ids, list):
        for item in ids:
            if isinstance(item, dict) and isinstance(item.get("value"), str):
                identifier = item["value"]
                break
    family = ""
    given = ""
    names = row.get("name")
    if isinstance(names, list):
        for item in names:
            if isinstance(item, dict):
                family = str(item.get("family") or "")
                values = item.get("given")
                if isinstance(values, list):
                    given = " ".join(str(v) for v in values if v)
                break
    return {
        "id": row.get("id"),
        "identifier": identifier,
        "name_family": family,
        "name_given": given,
        "birth_date": row.get("birthDate"),
        "gender": row.get("gender"),
    }


def flatten_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    concept = row.get("code")
    coding = first_coding(concept)
    quantity = row.get("valueQuantity")
    return {
        "id": row.get("id"),
        "patient_id": subject_id(row),
        "code": coding.get("code"),
        "code_text": coding.get("display")
        or (concept.get("text") if isinstance(concept, dict) else "")
        or "",
        "category": category_code(row),
        "effective_date_time": row.get("effectiveDateTime"),
        "value_quantity_value": (
            quantity.get("value") if isinstance(quantity, dict) else None
        ),
        "value_quantity_unit": (
            quantity.get("unit") if isinstance(quantity, dict) else None
        ),
        "value_string": row.get("valueString"),
        "status": row.get("status"),
    }


def flatten_condition(row: Mapping[str, Any]) -> dict[str, Any]:
    concept = row.get("code")
    coding = first_coding(concept)
    clinical = first_coding(row.get("clinicalStatus"))
    return {
        "id": row.get("id"),
        "patient_id": subject_id(row),
        "code_text": coding.get("display")
        or (concept.get("text") if isinstance(concept, dict) else "")
        or "",
        "clinical_status": clinical.get("code"),
        "category": category_code(row),
        "recorded_date": row.get("recordedDate"),
    }


def flatten_medication(row: Mapping[str, Any]) -> dict[str, Any]:
    concept = row.get("medicationCodeableConcept")
    coding = first_coding(concept)
    dosages = row.get("dosageInstruction")
    parts: list[str] = []
    if isinstance(dosages, list):
        for dosage in dosages:
            if not isinstance(dosage, dict):
                continue
            for key in ("text", "patientInstruction"):
                value = dosage.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)
            route = dosage.get("route")
            if isinstance(route, dict) and isinstance(route.get("text"), str):
                parts.append(route["text"])
            timing = dosage.get("timing")
            if isinstance(timing, dict):
                code = timing.get("code")
                if isinstance(code, dict) and isinstance(code.get("text"), str):
                    parts.append(code["text"])
            rates = dosage.get("doseAndRate")
            if isinstance(rates, list):
                for rate in rates:
                    dose = rate.get("doseQuantity") if isinstance(rate, dict) else None
                    if isinstance(dose, dict):
                        value, unit = dose.get("value"), dose.get("unit")
                        if value is not None or unit:
                            parts.append(f"{value or ''} {unit or ''}".strip())
    return {
        "id": row.get("id"),
        "patient_id": subject_id(row),
        "medication_text": concept.get("text") if isinstance(concept, dict) else "",
        "medication_code": coding.get("code"),
        "authored_on": row.get("authoredOn"),
        "status": row.get("status"),
        "intent": row.get("intent"),
        "dosage_text": "; ".join(dict.fromkeys(parts)),
    }


def flatten_procedure(row: Mapping[str, Any]) -> dict[str, Any]:
    concept = row.get("code")
    coding = first_coding(concept)
    return {
        "id": row.get("id"),
        "patient_id": subject_id(row),
        "code": coding.get("code"),
        "code_text": coding.get("display")
        or (concept.get("text") if isinstance(concept, dict) else "")
        or "",
        "performed_date_time": row.get("performedDateTime"),
        "status": row.get("status"),
    }


FLATTENERS = {
    "patient": flatten_patient,
    "observation": flatten_observation,
    "condition": flatten_condition,
    "medication_request": flatten_medication,
    "procedure": flatten_procedure,
}


def copy_value(value: Any) -> str:
    if value is None:
        return "\\N"
    if isinstance(value, bool):
        return "t" if value else "f"
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def write_copy_file(table: str, resources: Iterator[dict[str, Any]]) -> int:
    path = WORK_DIR / f"{table}.tsv"
    columns = TABLE_COLUMNS[table]
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for resource in resources:
            row = FLATTENERS[table](resource)
            writer.writerow([copy_value(row.get(column)) for column in columns])
            count += 1
    log(f"{table}: wrote {count} rows")
    return count


def copy_table(db: DatabaseConfig, table: str) -> None:
    path = WORK_DIR / f"{table}.tsv"
    columns = ", ".join(TABLE_COLUMNS[table])
    sql = (
        f"TRUNCATE fhir.{table}; COPY fhir.{table} ({columns}) FROM STDIN "
        "WITH (FORMAT csv, DELIMITER E'\\t', NULL E'\\\\N')"
    )
    with path.open("rb") as handle:
        subprocess.run(
            [
                "psql",
                "-h",
                db.host,
                "-p",
                db.port,
                "-U",
                db.user,
                "-d",
                db.name,
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                sql,
            ],
            check=True,
            stdin=handle,
            env=psql_env(db),
        )


def search_count(url: str) -> int:
    parts = urlsplit(url)
    params = parse_qs(parts.query, keep_blank_values=True)
    params.pop("_format", None)
    params["_summary"] = ["count"]
    params["_format"] = ["json"]
    resource = parts.path.rsplit("/", 1)[-1]
    bundle = fetch_bundle(f"{FHIR_BASE}/{resource}?{urlencode(params, doseq=True)}")
    total = bundle.get("total")
    if not isinstance(total, int):
        raise RuntimeError(f"FHIR count response has no integer total: {url}")
    return total


def verify_searches(db: DatabaseConfig) -> list[tuple[str, int, int]]:
    urls = [
        "http://localhost:8080/fhir/Patient?name=Maria&birthdate=1940-03-05",
        "http://localhost:8080/fhir/Patient?identifier=S6426560",
        "http://localhost:8080/fhir/Observation?patient=S6426560&code=MG",
        "http://localhost:8080/fhir/Observation?patient=S6426560&category=vital-signs",
        "http://localhost:8080/fhir/Observation?patient=S6426560&code=MG&date=ge2023-10-01",
        "http://localhost:8080/fhir/Observation?patient=S6426560&code=MG&date=lt2023-11-01",
        "http://localhost:8080/fhir/Condition?patient=S6426560&category=problem-list-item",
        "http://localhost:8080/fhir/Condition?patient=S6426560",
        "http://localhost:8080/fhir/MedicationRequest?patient=S6426560",
        "http://localhost:8080/fhir/Procedure?patient=S6426560&date=ge2023-10-01",
        "http://localhost:8080/fhir/Procedure?patient=S6426560&code=NUR1044",
        "http://localhost:8080/fhir/ServiceRequest?patient=S6426560",
    ]
    result: list[tuple[str, int, int]] = []
    for url in urls:
        translated = translate_search(url)
        store_count = search_count(url)
        sql_count = int(
            psql(db, f"SELECT count(*) FROM ({translated.sql}) AS q").strip()
        )
        result.append((url, store_count, sql_count))
    return result


def table_row_count(db: DatabaseConfig, table: str) -> int:
    return int(psql(db, f"SELECT count(*) FROM fhir.{table}").strip())


def write_record(
    *,
    started: datetime,
    finished: datetime,
    mrns: list[str],
    counts: Mapping[str, int],
    comparisons: list[tuple[str, int, int]],
    elapsed: float,
) -> None:
    lines = [
        "# MedAgentBench FHIR load run record",
        "",
        f"- Started: {started.astimezone(SGT).isoformat(timespec='seconds')}",
        f"- Finished: {finished.astimezone(SGT).isoformat(timespec='seconds')}",
        f"- Wall clock: {elapsed:.1f} seconds",
        f"- eval_MRN count: {len(mrns)}",
        "",
        "## Schema",
        "",
        "Schema: fhir. Indexed every patient_id column and every resource date column.",
        "",
        "## Row counts",
        "",
        "| Table | Rows |",
        "|---|---:|",
    ]
    for table, count in counts.items():
        lines.append(f"| fhir.{table} | {count} |")
    lines += [
        "",
        "## Search verification",
        "",
        "| FHIR search | Store count | SQL count | Match |",
        "|---|---:|---:|---|",
    ]
    for url, store_count, sql_count in comparisons:
        lines.append(
            f"| {url} | {store_count} | {sql_count} | "
            f"{'yes' if store_count == sql_count else 'NO'} |"
        )
    RECORD_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.touch()
    started = datetime.now(UTC)
    monotonic = time.monotonic()
    db = read_db_config()
    log("starting MedAgentBench FHIR load")
    mrns = load_mrns()
    log(f"loaded {len(mrns)} distinct eval_MRN values")
    psql(db, SCHEMA_SQL)
    log("created or verified fhir schema and indexes")
    counts: dict[str, int] = {}
    for table, resource in {
        "patient": "Patient",
        "observation": "Observation",
        "condition": "Condition",
        "medication_request": "MedicationRequest",
        "procedure": "Procedure",
    }.items():
        counts[table] = write_copy_file(table, iter_resources(resource, mrns))
        copy_table(db, table)
    psql(db, "TRUNCATE fhir.service_request")
    counts["service_request"] = 0
    actual = {table: table_row_count(db, table) for table in TABLE_COLUMNS}
    if actual != counts:
        raise RuntimeError(f"row count mismatch: wrote {counts}, loaded {actual}")
    comparisons = verify_searches(db)
    mismatches = sum(store != sql for _, store, sql in comparisons)
    finished = datetime.now(UTC)
    write_record(
        started=started,
        finished=finished,
        mrns=mrns,
        counts=actual,
        comparisons=comparisons,
        elapsed=time.monotonic() - monotonic,
    )
    log(f"finished; matched {len(comparisons) - mismatches}/{len(comparisons)}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
