from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlsplit

_RESOURCE_TABLES = {
    "Patient": "fhir.patient",
    "Observation": "fhir.observation",
    "Condition": "fhir.condition",
    "MedicationRequest": "fhir.medication_request",
    "Procedure": "fhir.procedure",
    "ServiceRequest": "fhir.service_request",
}

_DATE_COLUMNS = {
    "Patient": "birth_date",
    "Observation": "effective_date_time",
    "Condition": "recorded_date",
    "MedicationRequest": "authored_on",
    "Procedure": "performed_date_time",
    "ServiceRequest": "authored_on",
}

_ALLOWED_PARAMS = frozenset(
    {
        "patient",
        "code",
        "category",
        "date",
        "_id",
        "identifier",
        "name",
        "family",
        "given",
        "birthdate",
        "_format",
        "_count",
        "_sort",
    }
)


class FhirSearchRejected(ValueError):
    """A FHIR search URL that cannot be translated deterministically."""


@dataclass(frozen=True)
class SearchSql:
    """SQL text and the flattened FHIR resource table it reads."""

    sql: str
    table: str


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _patient_id(value: str) -> str:
    prefix = "Patient/"
    return value[len(prefix) :] if value.startswith(prefix) else value


def _date_period(body: str) -> tuple[date, date]:
    parts = body.split("-")
    if len(parts) == 1:
        return date(int(parts[0]), 1, 1), date(int(parts[0]) + 1, 1, 1)
    if len(parts) == 2:
        start = date(int(parts[0]), int(parts[1]), 1)
    else:
        start = date.fromisoformat(body)
    if len(parts) == 3:
        return start, start + timedelta(days=1)
    if start.month == 12:
        return start, date(start.year + 1, 1, 1)
    return start, date(start.year, start.month + 1, 1)


def _stamp(day: date) -> str:
    return _quote(day.isoformat() + "T00:00:00+00:00")


def _instant(body: str) -> str:
    text = body.replace(" ", "+")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).isoformat()


def _date_clause(value: str, column: str) -> str:
    prefix = next(
        (
            p
            for p in ("ge", "gt", "le", "lt")
            if value.startswith(p) and len(value) > len(p)
        ),
        "",
    )
    body = value[len(prefix) :] if prefix else value
    if "T" in body:
        operator = {"": "=", "ge": ">=", "gt": ">", "le": "<=", "lt": "<"}[prefix]
        return f"{column} {operator} {_quote(_instant(body))}"
    start, end = _date_period(body)
    if not prefix:
        return f"{column} >= {_stamp(start)} AND {column} < {_stamp(end)}"
    if prefix == "ge":
        return f"{column} >= {_stamp(end - timedelta(days=1))}"
    if prefix == "gt":
        return f"{column} >= {_stamp(end)}"
    if prefix == "le":
        return f"{column} < {_stamp(start + timedelta(days=1))}"
    return f"{column} < {_stamp(start)}"


def translate_search(url: str) -> SearchSql:
    """Translate one MedAgentBench FHIR search URL into SQL."""
    if not isinstance(url, str) or not url.strip():
        raise FhirSearchRejected("FHIR search URL is empty")
    parts = urlsplit(url.strip())
    path_parts = [part for part in parts.path.split("/") if part]
    if not path_parts:
        raise FhirSearchRejected("FHIR search URL does not name a resource")
    resource = path_parts[-1]
    if resource not in _RESOURCE_TABLES:
        raise FhirSearchRejected(f"unsupported FHIR resource {resource!r}")

    params: dict[str, str] = {}
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key not in _ALLOWED_PARAMS:
            raise FhirSearchRejected(f"unsupported FHIR search parameter {key!r}")
        if key in params and key != "_format":
            raise FhirSearchRejected(f"duplicate FHIR search parameter {key!r}")
        params[key] = value
    params.pop("_format", None)

    if resource != "Patient" and "patient" not in params:
        raise FhirSearchRejected(f"{resource} search requires patient")
    if resource == "Patient" and not any(
        key in params
        for key in ("_id", "identifier", "name", "family", "given", "birthdate")
    ):
        raise FhirSearchRejected(
            "Patient search requires _id, identifier, name, family, given or birthdate"
        )

    clauses: list[str] = []
    table = _RESOURCE_TABLES[resource]
    select = "*"
    if resource == "Patient":
        unsupported = set(params) & {"patient", "code", "category", "date"}
        if unsupported:
            key = sorted(unsupported)[0]
            raise FhirSearchRejected(f"unsupported Patient search parameter {key!r}")
        if "_id" in params:
            clauses.append(f"id = {_quote(_patient_id(params['_id']))}")
        if "identifier" in params:
            clauses.append(f"identifier = {_quote(params['identifier'])}")
        if "name" in params:
            clauses.append(
                f"(name_family ILIKE {_quote('%' + params['name'] + '%')} "
                f"OR name_given ILIKE {_quote('%' + params['name'] + '%')})"
            )
        if "family" in params:
            if not params["family"].strip():
                raise FhirSearchRejected("FHIR search parameter 'family' is blank")
            clauses.append(f"name_family ILIKE {_quote('%' + params['family'] + '%')}")
        if "given" in params:
            if not params["given"].strip():
                raise FhirSearchRejected("FHIR search parameter 'given' is blank")
            clauses.append(f"name_given ILIKE {_quote('%' + params['given'] + '%')}")
        if "birthdate" in params:
            clauses.append(f"birth_date = {_quote(params['birthdate'])}")
        if "name" in params and not any(key in params for key in ("_id", "identifier")):
            select = "id, identifier"
    else:
        unsupported = set(params) & {
            "_id",
            "identifier",
            "name",
            "family",
            "given",
            "birthdate",
        }
        if unsupported:
            key = sorted(unsupported)[0]
            raise FhirSearchRejected(f"unsupported {resource} search parameter {key!r}")
        clauses.append(f"patient_id = {_quote(_patient_id(params['patient']))}")

    if "code" in params:
        code_columns = {
            "Observation": "code",
            "Condition": "code_text",
            "MedicationRequest": "medication_code",
            "Procedure": "code",
            "ServiceRequest": "code",
        }
        if resource not in code_columns:
            raise FhirSearchRejected(f"unsupported {resource} search parameter 'code'")
        clauses.append(f"{code_columns[resource]} = {_quote(params['code'])}")
    if "category" in params:
        if resource not in {"Observation", "Condition", "MedicationRequest"}:
            raise FhirSearchRejected(
                f"unsupported {resource} search parameter 'category'"
            )
        clauses.append(f"category = {_quote(params['category'])}")
    if "date" in params:
        clauses.append(_date_clause(params["date"], _DATE_COLUMNS[resource]))

    order_by: str | None = None
    if "_sort" in params:
        raw_sort = params["_sort"]
        descending = raw_sort.startswith("-")
        sort_key = raw_sort[1:] if descending else raw_sort
        if sort_key not in {"date", "_lastUpdated"}:
            raise FhirSearchRejected(f"unsupported FHIR sort key {sort_key!r}")
        order_by = (
            f"ORDER BY {_DATE_COLUMNS[resource]} " f"{'DESC' if descending else 'ASC'}"
        )
    elif resource in {"Observation", "Procedure"}:
        order_by = f"ORDER BY {_DATE_COLUMNS[resource]} DESC"

    limit: str | None = None
    if "_count" in params:
        raw_count = params["_count"]
        if not raw_count.isdigit() or int(raw_count) < 1:
            raise FhirSearchRejected(f"invalid _count {raw_count!r}")
        limit = f"LIMIT {int(raw_count)}"

    sql = f"SELECT {select} FROM {table}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order_by:
        sql += f" {order_by}"
    if limit:
        sql += f" {limit}"
    return SearchSql(sql=sql, table=table)


def _subject(row: Mapping[str, Any]) -> dict[str, Any] | None:
    patient_id = row.get("patient_id")
    if patient_id in (None, ""):
        return None
    return {"reference": f"Patient/{_patient_id(str(patient_id))}"}


def _codeable(
    row: Mapping[str, Any], code_key: str = "code", text_key: str = "code_text"
) -> dict[str, Any] | None:
    code = row.get(code_key)
    text = row.get(text_key)
    result: dict[str, Any] = {}
    if code not in (None, ""):
        result["coding"] = [{"code": str(code)}]
    if text not in (None, ""):
        if "coding" in result:
            result["coding"][0]["display"] = str(text)
        result["text"] = str(text)
    return result or None


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        target[key] = value


def _render_patient(row: Mapping[str, Any]) -> dict[str, Any]:
    resource: dict[str, Any] = {"resourceType": "Patient"}
    _put(resource, "id", row.get("id"))
    identifier = row.get("identifier")
    if identifier not in (None, ""):
        resource["identifier"] = [{"value": identifier}]
    name: dict[str, Any] = {}
    _put(name, "family", row.get("name_family"))
    given = row.get("name_given")
    if given not in (None, ""):
        name["given"] = [given]
    if name:
        resource["name"] = [name]
    _put(resource, "birthDate", row.get("birth_date"))
    _put(resource, "gender", row.get("gender"))
    return resource


def _render_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    resource: dict[str, Any] = {"resourceType": "Observation"}
    _put(resource, "id", row.get("id"))
    _put(resource, "status", row.get("status"))
    _put(resource, "subject", _subject(row))
    _put(resource, "code", _codeable(row))
    category = row.get("category")
    if category not in (None, ""):
        resource["category"] = [{"coding": [{"code": category}]}]
    _put(resource, "effectiveDateTime", row.get("effective_date_time"))
    value = row.get("value_quantity_value")
    unit = row.get("value_quantity_unit")
    if value is not None:
        quantity: dict[str, Any] = {"value": value}
        _put(quantity, "unit", unit)
        resource["valueQuantity"] = quantity
    elif row.get("value_string") not in (None, ""):
        resource["valueString"] = row["value_string"]
    return resource


def _render_condition(row: Mapping[str, Any]) -> dict[str, Any]:
    resource: dict[str, Any] = {"resourceType": "Condition"}
    _put(resource, "id", row.get("id"))
    _put(resource, "subject", _subject(row))
    _put(resource, "code", _codeable(row, "code", "code_text"))
    clinical = row.get("clinical_status")
    if clinical not in (None, ""):
        resource["clinicalStatus"] = {"coding": [{"code": clinical}]}
    category = row.get("category")
    if category not in (None, ""):
        resource["category"] = [{"coding": [{"code": category}]}]
    _put(resource, "recordedDate", row.get("recorded_date"))
    return resource


def _render_medication_request(row: Mapping[str, Any]) -> dict[str, Any]:
    resource: dict[str, Any] = {"resourceType": "MedicationRequest"}
    _put(resource, "id", row.get("id"))
    _put(resource, "status", row.get("status"))
    _put(resource, "intent", row.get("intent"))
    _put(resource, "subject", _subject(row))
    _put(
        resource,
        "medicationCodeableConcept",
        _codeable(row, "medication_code", "medication_text"),
    )
    _put(resource, "authoredOn", row.get("authored_on"))
    dosage = row.get("dosage_text")
    if dosage not in (None, ""):
        resource["dosageInstruction"] = [{"text": dosage}]
    return resource


def _render_procedure(row: Mapping[str, Any]) -> dict[str, Any]:
    resource: dict[str, Any] = {"resourceType": "Procedure"}
    _put(resource, "id", row.get("id"))
    _put(resource, "status", row.get("status"))
    _put(resource, "subject", _subject(row))
    _put(resource, "code", _codeable(row))
    _put(resource, "performedDateTime", row.get("performed_date_time"))
    return resource


def _render_service_request(row: Mapping[str, Any]) -> dict[str, Any]:
    resource: dict[str, Any] = {"resourceType": "ServiceRequest"}
    _put(resource, "id", row.get("id"))
    _put(resource, "status", row.get("status"))
    _put(resource, "intent", row.get("intent"))
    _put(resource, "priority", row.get("priority"))
    _put(resource, "subject", _subject(row))
    _put(resource, "code", _codeable(row))
    _put(resource, "authoredOn", row.get("authored_on"))
    _put(resource, "occurrenceDateTime", row.get("occurrence_date_time"))
    note = row.get("note_text")
    if note not in (None, ""):
        resource["note"] = [{"text": note}]
    return resource


_RENDERERS = {
    "Patient": _render_patient,
    "Observation": _render_observation,
    "Condition": _render_condition,
    "MedicationRequest": _render_medication_request,
    "Procedure": _render_procedure,
    "ServiceRequest": _render_service_request,
}


def render_bundle(resource: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render flattened SQL rows as the HAPI searchset bundle the model saw."""
    renderer = _RENDERERS.get(resource)
    if renderer is None:
        raise FhirSearchRejected(f"unsupported FHIR resource {resource!r}")
    rendered_rows = [renderer(row) for row in rows]
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(rendered_rows),
        "entry": [{"resource": row} for row in rendered_rows],
    }
