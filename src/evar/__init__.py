"""EVAR: Evidence-Validated Hypothesis Admission (reference implementation).

Independent implementation of the admission loop described by
Liu, Ji, Ping. EVAR: Evidence-Validated Hypothesis Admission for
Budget-Aware Narrative Reasoning. arXiv:2608.29835, 30 Aug 2026. EMNLP 2026.

This is not the authors' official code. The default reasoner is deterministic.
"""

from evar.engine import EVAREngine, run_evar
from evar.store import EvidenceStore
from evar.types import (
    AtomicClaim,
    Challenge,
    EVARResult,
    FrozenStoreError,
    Gap,
    Hypothesis,
    RoundLog,
    SourceSpan,
    Verdict,
)

__all__ = [
    "EVAREngine",
    "EvidenceStore",
    "Verdict",
    "run_evar",
    "FrozenStoreError",
    "AtomicClaim",
    "Challenge",
    "EVARResult",
    "Gap",
    "Hypothesis",
    "RoundLog",
    "SourceSpan",
]

__version__ = "0.1.0"
