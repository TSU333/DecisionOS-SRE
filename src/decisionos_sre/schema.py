from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)

class Candidate(Strict):
    candidate_id: str = Field(min_length=1, max_length=300)
    display_name: str = Field(min_length=1, max_length=300)
    observable_description: str = Field(default="", max_length=1000)

class TemporalSummary(Strict):
    q10_z: float = Field(ge=-100, le=100)
    q90_z: float = Field(ge=-100, le=100)
    std_ratio: float = Field(ge=0, le=100)
    trend_z: float = Field(ge=-100, le=100)
    late_shift_z: float = Field(ge=-100, le=100)

class Metric(Strict):
    service: str = Field(min_length=1)
    name: str = Field(min_length=1)
    unit: str = "source_unit_unspecified"
    baseline_mean: float | None = None
    observed_mean: float | None = None
    change_z: float | None = None
    missing_fraction: float = Field(ge=0, le=1)
    baseline_samples: int = Field(ge=0)
    observed_samples: int = Field(ge=0)
    observed_until: float
    temporal: TemporalSummary | None = None

class Evidence(Strict):
    metrics: list[Metric] = Field(default_factory=list, max_length=5000)
    logs: None = None
    traces: None = None
    topology: None = None
    deployment_context: None = None

class Availability(Strict):
    metrics: bool
    logs: Literal[False] = False
    traces: Literal[False] = False
    topology: Literal[False] = False
    deployment_context: Literal[False] = False

class IncidentInput(Strict):
    application: str = Field(min_length=1, max_length=200)
    decision_time: float
    candidates: list[Candidate] = Field(min_length=1, max_length=1000)
    evidence: Evidence
    modality_availability: Availability

    @model_validator(mode="after")
    def validate_incident(self):
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate_id")
        if any(m.observed_until > self.decision_time for m in self.evidence.metrics):
            raise ValueError("evidence after decision_time")
        if self.modality_availability.metrics != bool(self.evidence.metrics):
            raise ValueError("metrics availability disagrees with evidence")
        return self

class Provenance(Strict):
    kind: Literal["gold", "policy_derived", "teacher"]
    source_ref: str
    mapping_version: str
    collection_method: str

class Target(Strict):
    value: str | None
    raw_value: str | list[str] | None
    provenance: Provenance

class Targets(Strict):
    root_cause: Target
    fault_type: Target

class TrainingExample(Strict):
    schema_version: Literal["1"] = "1"
    opaque_incident_id: str
    original_run_id: str
    parent_incident_id: str
    source_metadata: dict
    input: IncidentInput
    targets: Targets
    augmentation_metadata: dict = Field(default_factory=dict)

class TaskDecision(Strict):
    selected: str
    probabilities: dict[str, float]
    confidence: float
    calibration_status: str

class Routing(Strict):
    destination: Literal["ACCEPT_DIAGNOSIS", "REVIEW"]
    routing_score: float | None
    reason_codes: list[str]

class DecisionResponse(Strict):
    root_cause: TaskDecision | None
    fault_type: TaskDecision | None
    routing: Routing
    versions: dict[str, str | None]
    evidence_status: dict
    timings_ms: dict[str, float]
    execute_remediation: Literal[False] = False
