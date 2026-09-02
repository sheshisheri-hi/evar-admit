"""Frozen datatypes for the EVAR admission loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Polarity(str, Enum):
    ASSERTED = "asserted"
    NEGATED = "negated"


class ClaimStatus(str, Enum):
    OK = "OK"
    UNCERTAIN = "Uncertain"
    CONFLICT = "Conflict"


class Verdict(str, Enum):
    ADMITTED = "ADMITTED"
    QUARANTINED = "QUARANTINED"
    DISCARDED = "DISCARDED"


class StopReason(str, Enum):
    EMPTY_INPUT = "empty_input"
    SUFFICIENCY = "sufficiency"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_CANDIDATES = "no_candidates"
    FAST_PATH = "fast_path"


@dataclass(frozen=True)
class SourceSpan:
    """Character offsets into the original narrative."""

    start: int
    end: int
    text: str

    def slice(self, narrative: str) -> str:
        return narrative[self.start : self.end]


@dataclass(frozen=True)
class AtomicClaim:
    """One source-linked atomic proposition compiled from the narrative."""

    id: str
    text: str
    source: SourceSpan
    entities: tuple[str, ...]
    polarity: Polarity
    predicates: tuple[str, ...]
    status: ClaimStatus = ClaimStatus.OK
    severity: int = 0
    note: str = ""
    times: tuple[str, ...] = ()
    places: tuple[str, ...] = ()
    is_rumor: bool = False


@dataclass(frozen=True)
class Gap:
    """An underspecified premise that currently blocks a grounded answer."""

    id: str
    slot: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class Hypothesis:
    """A candidate explanation proposed for an unresolved gap.

    ``fillers`` maps slot name -> filler string (as a tuple of pairs so the
    dataclass can stay frozen).
    """

    id: str
    statement: str
    target_gap: str
    entities: tuple[str, ...]
    predicates: tuple[str, ...]
    polarity: Polarity = Polarity.ASSERTED
    slot: str | None = None
    fillers: tuple[tuple[str, str], ...] = ()

    def filler_map(self) -> Mapping[str, str]:
        return dict(self.fillers)


@dataclass(frozen=True)
class Challenge:
    """Hypothesis-conditioned validation challenge (not itself evidence)."""

    hypothesis_id: str
    support_queries: tuple[str, ...]
    contradiction_queries: tuple[str, ...]
    prerequisite_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerdictRecord:
    hypothesis_id: str
    verdict: Verdict
    support_score: float
    contradicted: bool
    supporting_claim_ids: tuple[str, ...] = ()
    contradicting_claim_ids: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class RoundLog:
    round_index: int
    proposed: tuple[Hypothesis, ...]
    challenges: tuple[Challenge, ...]
    verdicts: tuple[VerdictRecord, ...]
    remaining_rounds: int
    remaining_hypotheses: int
    unresolved_slots: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class EVARResult:
    claims: tuple[AtomicClaim, ...]
    budget_max_rounds: int
    budget_max_hypotheses: int
    n_uncertain: int
    rounds: tuple[RoundLog, ...]
    admitted: tuple[Hypothesis, ...]
    quarantined: tuple[Hypothesis, ...]
    discarded: tuple[Hypothesis, ...]
    answer: str
    stop_reason: str
    covered_slots: tuple[str, ...]
    unresolved_slots: tuple[str, ...]
    question: str = ""
    narrative: str = ""


class FrozenStoreError(RuntimeError):
    """Raised when a caller tries to mutate a locked evidence store."""


class EVARError(ValueError):
    """Clean, non-traceback-oriented input/usage error."""
