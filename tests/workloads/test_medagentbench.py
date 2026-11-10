import pytest

from lumilake_server.workloads.medagentbench import (
    FhirSearchRejected,
    render_bundle,
    translate_search,
)


def test_patient_name_and_birthdate_translation() -> None:
    translated = translate_search(
        "http://localhost:8080/fhir/Patient?name=Peter%20Stafford&birthdate=1932-12-29"
    )
    assert translated.table == "fhir.patient"
    assert translated.sql == (
        "SELECT id, identifier FROM fhir.patient WHERE "
        "(name_family ILIKE '%Peter Stafford%' OR name_given ILIKE '%Peter Stafford%') "
        "AND birth_date = '1932-12-29'"
    )


def test_patient_id_and_identifier_translation() -> None:
    assert translate_search("Patient?_id=Patient%2Fp1").sql == (
        "SELECT * FROM fhir.patient WHERE id = 'p1'"
    )
    assert translate_search("Patient?identifier=S6534835").sql == (
        "SELECT * FROM fhir.patient WHERE identifier = 'S6534835'"
    )


def test_observation_code_and_ge_date_translation() -> None:
    translated = translate_search(
        "Observation?patient=123&code=MG&date=ge2023-11-13&_format=json"
    )
    assert translated.table == "fhir.observation"
    assert translated.sql == (
        "SELECT * FROM fhir.observation WHERE patient_id = '123' "
        "AND code = 'MG' AND effective_date_time >= '2023-11-13T00:00:00+00:00' "
        "ORDER BY id::bigint ASC"
    )


def test_observation_category_translation() -> None:
    assert translate_search("Observation?patient=123&category=vital-signs").sql == (
        "SELECT * FROM fhir.observation WHERE patient_id = '123' "
        "AND category = 'vital-signs' ORDER BY id::bigint ASC"
    )


def test_condition_category_translation() -> None:
    assert translate_search("Condition?patient=123&category=problem-list-item").sql == (
        "SELECT * FROM fhir.condition WHERE patient_id = '123' "
        "AND category = 'problem-list-item' ORDER BY id::bigint ASC"
    )


def test_medication_request_translation() -> None:
    assert translate_search(
        "MedicationRequest?patient=123&category=Community&date=ge2023-11-13"
    ).sql == (
        "SELECT * FROM fhir.medication_request WHERE patient_id = '123' "
        "AND category = 'Community' AND authored_on >= '2023-11-13T00:00:00+00:00' "
        "ORDER BY id::bigint ASC"
    )


def test_procedure_translation() -> None:
    assert translate_search(
        "Procedure?patient=123&date=ge2023-11-13&code=12345"
    ).sql == (
        "SELECT * FROM fhir.procedure WHERE patient_id = '123' "
        "AND code = '12345' AND performed_date_time >= '2023-11-13T00:00:00+00:00' "
        "ORDER BY id::bigint ASC"
    )


def test_service_request_search_is_accepted() -> None:
    assert translate_search("ServiceRequest?patient=123&code=93000").sql == (
        "SELECT * FROM fhir.service_request WHERE patient_id = '123' "
        "AND code = '93000' ORDER BY id::bigint ASC"
    )


def test_count_and_sort_translation() -> None:
    assert translate_search(
        "Observation?patient=123&code=MG&_count=5&_sort=-date"
    ).sql == (
        "SELECT * FROM fhir.observation WHERE patient_id = '123' AND code = 'MG' "
        "ORDER BY effective_date_time DESC LIMIT 5"
    )


def test_last_updated_sort_translation() -> None:
    assert translate_search("Condition?patient=123&_sort=_lastUpdated").sql == (
        "SELECT * FROM fhir.condition WHERE patient_id = '123' "
        "ORDER BY recorded_date ASC"
    )


def test_default_order_is_ascending_numeric_id() -> None:
    assert translate_search("Observation?patient=123&code=MG").sql == (
        "SELECT * FROM fhir.observation WHERE patient_id = '123' AND code = 'MG' "
        "ORDER BY id::bigint ASC"
    )
    assert translate_search("Condition?patient=123").sql == (
        "SELECT * FROM fhir.condition WHERE patient_id = '123' ORDER BY id::bigint ASC"
    )
    assert translate_search("MedicationRequest?patient=123").sql == (
        "SELECT * FROM fhir.medication_request WHERE patient_id = '123' "
        "ORDER BY id::bigint ASC"
    )
    assert translate_search("Procedure?patient=123").sql == (
        "SELECT * FROM fhir.procedure WHERE patient_id = '123' ORDER BY id::bigint ASC"
    )


def test_sort_by_id_keeps_the_numeric_order() -> None:
    assert translate_search("Observation?patient=123&_sort=_id").sql == (
        "SELECT * FROM fhir.observation WHERE patient_id = '123' "
        "ORDER BY id::bigint ASC"
    )
    assert translate_search("Observation?patient=123&_sort=-_id").sql == (
        "SELECT * FROM fhir.observation WHERE patient_id = '123' "
        "ORDER BY id::bigint DESC"
    )
    assert translate_search("Patient?_id=Patient%2Fp1&_sort=_id").sql == (
        "SELECT * FROM fhir.patient WHERE id = 'p1' ORDER BY id ASC"
    )


@pytest.mark.parametrize(
    ("value", "clause"),
    [
        (
            "2023-11-12T10:15:00+00:00",
            "effective_date_time = '2023-11-12T10:15:00+00:00'",
        ),
        (
            "2023-11-12T10%3A15%3A00%2B00%3A00",
            "effective_date_time = '2023-11-12T10:15:00+00:00'",
        ),
        ("2023-10-10T20:54:00Z", "effective_date_time = '2023-10-10T20:54:00+00:00'"),
        (
            "ge2023-11-12T10:15:00+00:00",
            "effective_date_time >= '2023-11-12T10:15:00+00:00'",
        ),
        (
            "gt2023-11-12T10:15:00+00:00",
            "effective_date_time > '2023-11-12T10:15:00+00:00'",
        ),
        (
            "le2023-11-12T10:15:00+00:00",
            "effective_date_time <= '2023-11-12T10:15:00+00:00'",
        ),
        (
            "lt2023-11-12T10:15:00+00:00",
            "effective_date_time < '2023-11-12T10:15:00+00:00'",
        ),
        (
            "2023-07-06",
            "effective_date_time >= '2023-07-06T00:00:00+00:00' "
            "AND effective_date_time < '2023-07-07T00:00:00+00:00'",
        ),
        (
            "2023-07",
            "effective_date_time >= '2023-07-01T00:00:00+00:00' "
            "AND effective_date_time < '2023-08-01T00:00:00+00:00'",
        ),
        (
            "2023-12",
            "effective_date_time >= '2023-12-01T00:00:00+00:00' "
            "AND effective_date_time < '2024-01-01T00:00:00+00:00'",
        ),
        (
            "2023",
            "effective_date_time >= '2023-01-01T00:00:00+00:00' "
            "AND effective_date_time < '2024-01-01T00:00:00+00:00'",
        ),
        ("ge2023-07", "effective_date_time >= '2023-07-31T00:00:00+00:00'"),
        ("gt2023-07", "effective_date_time >= '2023-08-01T00:00:00+00:00'"),
        ("le2023-07", "effective_date_time < '2023-07-02T00:00:00+00:00'"),
        ("lt2023-07", "effective_date_time < '2023-07-01T00:00:00+00:00'"),
        ("ge2023", "effective_date_time >= '2023-12-31T00:00:00+00:00'"),
        ("le2023", "effective_date_time < '2023-01-02T00:00:00+00:00'"),
        ("lt2023", "effective_date_time < '2023-01-01T00:00:00+00:00'"),
        ("ge2023-12", "effective_date_time >= '2023-12-31T00:00:00+00:00'"),
        ("gt2023-12", "effective_date_time >= '2024-01-01T00:00:00+00:00'"),
    ],
)
def test_date_parameter_translation(value: str, clause: str) -> None:
    sql = translate_search(f"Observation?patient=123&date={value}").sql
    assert clause in sql


def test_quote_escaping() -> None:
    assert translate_search("Patient?identifier=S%27O%27Brien").sql == (
        "SELECT * FROM fhir.patient WHERE identifier = 'S''O''Brien'"
    )


def test_patient_discrete_name_parts_translation() -> None:
    assert translate_search("Patient?family=Smith").sql == (
        "SELECT * FROM fhir.patient WHERE name_family ILIKE '%Smith%'"
    )
    assert translate_search("Patient?given=Peter").sql == (
        "SELECT * FROM fhir.patient WHERE name_given ILIKE '%Peter%'"
    )
    assert translate_search(
        "Patient?given=Kyle&family=Jia&birthdate=1969-05-15&_format=json"
    ).sql == (
        "SELECT * FROM fhir.patient WHERE name_family ILIKE '%Jia%' "
        "AND name_given ILIKE '%Kyle%' AND birth_date = '1969-05-15'"
    )


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("", "empty"),
        ("Practitioner?patient=123", "unsupported FHIR resource"),
        ("Observation?patient=123&foo=bar", "unsupported FHIR search parameter"),
        ("Observation?code=MG", "requires patient"),
        ("Observation?patient=123&name=Peter", "unsupported Observation"),
        ("Observation?patient=123&family=Peter", "unsupported Observation"),
        ("Observation?patient=123&given=Peter", "unsupported Observation"),
        ("Patient", "Patient search requires"),
    ],
)
def test_rejections(url: str, message: str) -> None:
    with pytest.raises(FhirSearchRejected, match=message):
        translate_search(url)


def test_render_patient_bundle() -> None:
    rows = [
        {
            "id": "p1",
            "identifier": "S6534835",
            "name_family": "Stafford",
            "name_given": "Peter",
            "birth_date": "1932-12-29",
            "gender": "male",
        }
    ]
    assert render_bundle("Patient", rows) == {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 1,
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "identifier": [{"value": "S6534835"}],
                    "name": [{"family": "Stafford", "given": ["Peter"]}],
                    "birthDate": "1932-12-29",
                    "gender": "male",
                }
            }
        ],
    }


def test_render_observation_bundle() -> None:
    rows = [
        {
            "id": "o1",
            "patient_id": "p1",
            "status": "final",
            "code": "MG",
            "code_text": "Magnesium",
            "category": "laboratory",
            "effective_date_time": "2023-11-13T09:00:00+00:00",
            "value_quantity_value": 2.1,
            "value_quantity_unit": "mg/dL",
        }
    ]
    assert render_bundle("Observation", rows) == {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 1,
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "o1",
                    "status": "final",
                    "subject": {"reference": "Patient/p1"},
                    "code": {
                        "coding": [{"code": "MG", "display": "Magnesium"}],
                        "text": "Magnesium",
                    },
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": "2023-11-13T09:00:00+00:00",
                    "valueQuantity": {"value": 2.1, "unit": "mg/dL"},
                }
            }
        ],
    }


def test_render_medication_request_bundle() -> None:
    rows = [
        {
            "id": "m1",
            "patient_id": "p1",
            "status": "active",
            "intent": "order",
            "medication_code": "123",
            "medication_text": "Aspirin",
            "authored_on": "2023-11-13",
            "dosage_text": "Take daily",
        }
    ]
    assert render_bundle("MedicationRequest", rows) == {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 1,
        "entry": [
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "m1",
                    "status": "active",
                    "intent": "order",
                    "subject": {"reference": "Patient/p1"},
                    "medicationCodeableConcept": {
                        "coding": [{"code": "123", "display": "Aspirin"}],
                        "text": "Aspirin",
                    },
                    "authoredOn": "2023-11-13",
                    "dosageInstruction": [{"text": "Take daily"}],
                }
            }
        ],
    }


def test_render_condition_procedure_and_service_request_bundles() -> None:
    condition = render_bundle(
        "Condition",
        [
            {
                "id": "c1",
                "patient_id": "p1",
                "code_text": "Hypertension",
                "clinical_status": "active",
                "category": "problem-list-item",
                "recorded_date": "2023-11-13",
            }
        ],
    )["entry"][0]["resource"]
    assert condition == {
        "resourceType": "Condition",
        "id": "c1",
        "subject": {"reference": "Patient/p1"},
        "code": {"text": "Hypertension"},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "category": [{"coding": [{"code": "problem-list-item"}]}],
        "recordedDate": "2023-11-13",
    }

    procedure = render_bundle(
        "Procedure",
        [{"id": "r1", "patient_id": "p1", "code": "12345", "code_text": "ECG"}],
    )["entry"][0]["resource"]
    assert procedure["resourceType"] == "Procedure"
    assert procedure["code"] == {
        "coding": [{"code": "12345", "display": "ECG"}],
        "text": "ECG",
    }

    service = render_bundle(
        "ServiceRequest",
        [
            {
                "id": "s1",
                "patient_id": "p1",
                "code": "93000",
                "code_text": "Referral",
                "status": "active",
                "intent": "order",
                "priority": "stat",
                "note_text": "Cardiology",
            }
        ],
    )["entry"][0]["resource"]
    assert service["resourceType"] == "ServiceRequest"
    assert service["note"] == [{"text": "Cardiology"}]


def test_empty_rows_render_an_empty_searchset() -> None:
    assert render_bundle("Observation", []) == {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 0,
        "entry": [],
    }
