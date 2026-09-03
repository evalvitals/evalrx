"""Wire contract for the M1-M5 diagnosis pipeline.

The single machine-readable definition of what crosses each stage boundary.
Python is the source of truth; JSON Schema and TypeScript are generated from it
(``python -m evalrx.contract.export``) so the three can never drift.

    from evalrx.contract import STAGE_IO, ProbeOutput

    ProbeOutput.model_validate_json(open("probe.json").read())   # backend gate
    STAGE_IO["m1"].output.model_json_schema()                    # frontend types

Scope: this describes the SERIALIZED shape, not the in-memory dataclasses. They
differ (``AnalysisReport.to_dict()`` drops ``raw_results``), and only the
serialized shape is what another process actually receives.

Reader-facing text
------------------
Every field a frontend puts on screen is written for a reader with no
background in this field -- assume a bright high-school student who has never
heard of an e-value, a tier, or a paired flip. That obligation sits here, on
the producer, and not on the frontend, for one reason: a consumer handed
``mcnemar_evalue`` or ``audio_evidence_then_answer`` can only guess, and a
guess dressed as a label ("Mcnemar Evalue") reads as though the run explained
itself when it did not.

So the contract separates the two jobs wherever they collide:

* an IDENTIFIER (``FixAttemptWire.name``, a signal's dotted path) joins records
  and must never change to suit a reader;
* a LABEL (``StatsToolResultWire.measured``, ``FixAttemptWire.ref``) is what a
  reader points at;
* a SENTENCE (``StatsToolResultWire.means``, ``FixAttemptWire.headline``,
  ``MethodologyWire.summary``) says what it means in plain language.

A producer with nothing readable to offer leaves the label and sentence empty.
An honest blank is a smaller error than a title-cased slug, because the blank
is visible and the slug is not.

Requires the ``contract`` extra: ``pip install evalrx[contract]``.
"""

from __future__ import annotations

from typing import NamedTuple

from evalrx.contract.common import (
    SCHEMA_VERSION,
    ArtifactRef,
    CaseBatchRef,
    CaseBatchWire,
    EvidenceGrade,
    ExternalRef,
    FailureCaseWire,
    HypothesisStatus,
    InputsWire,
    JoinReport,
    Label,
    MediaRef,
    Modality,
    ProvenanceWire,
    Source,
    StageEnvelope,
    StageState,
    StageStatus,
    StepWire,
    TrajectoryWire,
)
from evalrx.contract.m1 import (
    AnalyzerSelection,
    FindingsWire,
    ModelRef,
    PerCaseRow,
    ProbeInput,
    ProbeOutput,
    ProtocolWire,
    ResultWire,
)
from evalrx.contract.m2 import (
    AnalysisFindingWire,
    AnalysisInput,
    CorrectedRejections,
    ExploreContextWire,
    StatsReportWire,
    StatsToolResultWire,
)
from evalrx.contract.m3 import (
    DiagnosisInput,
    DiagnosisOutput,
    HypothesisWire,
)
from evalrx.contract.m4 import (
    FixAttemptWire,
    FixInput,
    FixOutput,
    InterventionOutput,
    SurgeryInput,
)
from evalrx.contract.m5 import (
    HypothesisTestInput,
    HypothesisTestOutput,
    HypothesisTestResultWire,
    TestEvidence,
)
from evalrx.contract.methodology import MethodologyWire
from evalrx.contract.pre_m1 import ProbeSearchInput, ProbeSearchOutput


class StageContract(NamedTuple):
    """One stage's input/output pair plus what it is for."""

    stage: str
    purpose: str
    input: type
    output: type
    optional: bool = False


#: The pipeline, as data. Docs, schema export and stage-map UIs all read this
#: instead of restating the wiring in prose that then drifts.
#:
#: Execution order note: the numbering is registration order, not run order.
#: ``VLDiagnoseLoop`` runs M1->M2->M3->M5 as its cycle and calls M4 once after,
#: because M4 is expensive and should only run on a hypothesis M5 verified.
STAGE_IO: dict[str, StageContract] = {
    "pre_m1": StageContract(
        "pre_m1", "Synthesize new failing cases (output is DATA, not a verdict).",
        ProbeSearchInput, ProbeSearchOutput, optional=True,
    ),
    "m1": StageContract(
        "m1", "Measure: run analyzers, emit per-case numbers. No pass/fail judgement.",
        ProbeInput, ProbeOutput,
    ),
    "m2": StageContract(
        "m2", "Screen: join numbers to labels, test which signals track failure, correct for multiplicity.",
        AnalysisInput, StatsReportWire,
    ),
    "m3": StageContract(
        "m3", "Explain: propose falsifiable mechanism hypotheses, each with a routable test_design.",
        DiagnosisInput, DiagnosisOutput,
    ),
    "m5": StageContract(
        "m5", "Adjudicate: statistical gate AND protocol-consistency gate. Both, or not SUPPORTED.",
        HypothesisTestInput, HypothesisTestOutput,
    ),
    "m4_surgery": StageContract(
        "m4_surgery", "Intervene: change one variable, re-run, read as causal evidence. Does not repair.",
        SurgeryInput, InterventionOutput,
    ),
    "m4_fix": StageContract(
        "m4_fix", "Repair: propose candidates, validate paired against the unmodified baseline.",
        FixInput, FixOutput, optional=True,
    ),
}

__all__ = [
    "SCHEMA_VERSION", "STAGE_IO", "StageContract",
    # common
    "ArtifactRef", "CaseBatchRef", "CaseBatchWire", "EvidenceGrade", "ExternalRef",
    "FailureCaseWire", "HypothesisStatus", "InputsWire", "JoinReport", "Label",
    "MediaRef", "Modality", "ProvenanceWire", "Source", "StageEnvelope", "StageState",
    "StageStatus", "StepWire", "TrajectoryWire",
    # pre-M1
    "ProbeSearchInput", "ProbeSearchOutput",
    # M1
    "ProbeInput", "ProbeOutput", "ResultWire", "FindingsWire", "PerCaseRow",
    "ModelRef", "ProtocolWire", "AnalyzerSelection",
    # M2
    "AnalysisInput", "StatsReportWire", "StatsToolResultWire", "AnalysisFindingWire",
    "CorrectedRejections", "ExploreContextWire",
    # M3
    "DiagnosisInput", "DiagnosisOutput", "HypothesisWire",
    # M4
    "SurgeryInput", "InterventionOutput",
    "FixInput", "FixOutput", "FixAttemptWire",
    # methodology
    "MethodologyWire",
    # M5
    "HypothesisTestInput", "HypothesisTestOutput", "HypothesisTestResultWire", "TestEvidence",
]
