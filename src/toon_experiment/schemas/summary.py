from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class Admission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[str] = None
    care_center_details: Optional[str] = Field(default=None, alias="care center details")
    details: Optional[str] = None


class PatientInformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    age: Optional[str] = None
    sex: Optional[str] = None
    ethnicity: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None
    family_medical_history: Optional[str] = Field(default=None, alias="family medical history")
    recent_travels: Optional[str] = Field(default=None, alias="recent travels")
    socio_economic_context: Optional[str] = Field(default=None, alias="socio economic context")
    occupation: Optional[str] = None


class PatientMedicalHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    physiological_context: Optional[str] = Field(default=None, alias="physiological context")
    psychological_context: Optional[str] = Field(default=None, alias="psychological context")
    vaccination_history: Optional[str] = Field(default=None, alias="vaccination history")
    allergies: Optional[str] = None
    exercise_frequency: Optional[str] = Field(default=None, alias="exercise frequency")
    nutrition: Optional[str] = None
    sexual_history: Optional[str] = Field(default=None, alias="sexual history")
    alcohol_consumption: Optional[str] = Field(default=None, alias="alcohol consumption")
    drug_usage: Optional[str] = Field(default=None, alias="drug usage")
    smoking_status: Optional[str] = Field(default=None, alias="smoking status")


class Surgery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: Optional[str] = None
    type: Optional[str] = Field(default=None, alias="Type")
    time: Optional[str] = None
    outcome: Optional[str] = None
    details: Optional[str] = None


class Symptom(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name_of_symptom: Optional[str] = Field(default=None, alias="name of symptom")
    intensity_of_symptom: Optional[str] = Field(default=None, alias="intensity of symptom")
    location: Optional[str] = None
    time: Optional[str] = None
    temporalisation: Optional[str] = None
    behaviours_affecting_symptom: Optional[str] = Field(default=None, alias="behaviours affecting the symptom")
    details: Optional[str] = None


class MedicalExamination(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    result: Optional[str] = None
    details: Optional[str] = None


class DiagnosticTest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    test: Optional[str] = None
    result: Optional[str] = None
    severity: Optional[str] = None
    condition: Optional[str] = None
    time: Optional[str] = None
    details: Optional[str] = None


class Treatment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    related_condition: Optional[str] = Field(default=None, alias="related condition")
    dosage: Optional[str] = None
    time: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    reason_for_taking: Optional[str] = Field(default=None, alias="reason for taking")
    reaction_to_treatment: Optional[str] = Field(default=None, alias="reaction to treatment")
    details: Optional[str] = None


class Discharge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: Optional[str] = None
    referral: Optional[str] = None
    follow_up: Optional[str] = Field(default=None, alias="follow up")
    discharge_summary: Optional[str] = Field(default=None, alias="discharge summary")


class Summary(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    visit_motivation: Optional[str] = Field(default=None, alias="visit motivation")
    admission: List[Admission] = Field(default_factory=list)
    patient_information: Optional[PatientInformation] = Field(default=None, alias="patient information")
    patient_medical_history: Optional[PatientMedicalHistory] = Field(default=None, alias="patient medical history")
    surgeries: List[Surgery] = Field(default_factory=list)
    symptoms: List[Symptom] = Field(default_factory=list)
    medical_examinations: List[MedicalExamination] = Field(default_factory=list, alias="medical examinations")
    diagnosis_tests: List[DiagnosticTest] = Field(default_factory=list, alias="diagnosis tests")
    treatments: List[Treatment] = Field(default_factory=list)
    discharge: Optional[Discharge] = Field(default=None)


def summary_template() -> dict:
    """Return a JSON-serializable template with all fields present."""
    template = Summary()
    return template.model_dump(mode="json", exclude_none=False, by_alias=True)
