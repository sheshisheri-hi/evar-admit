"""EVAREngine: compile → lock → budget → propose → challenge → verify → stop."""

from __future__ import annotations

import os
from collections.abc import Sequence

from evar.budget import Budget
from evar.compiler import compile_narrative
from evar.gaps import covered_slots, is_sufficient, parse_slots, unresolved_gaps
from evar.reasoner import Reasoner, build_reasoner
from evar.store import EvidenceStore
from evar.types import (
    EVARResult,
    Hypothesis,
    RoundLog,
    StopReason,
    Verdict,
)
from evar.validator import construct_challenge, validate

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from pydantic import BaseModel, Field


class EVARSettings(BaseModel):
    """Env-facing knobs. Default path stays fully offline."""

    reasoner: str = Field(default="deterministic")
    model: str = Field(default="gpt-4o-mini")

    @classmethod
    def from_env(cls) -> "EVARSettings":
        return cls(
            reasoner=os.getenv("EVAR_REASONER", "deterministic"),
            model=os.getenv("EVAR_MODEL", "gpt-4o-mini"),
        )


def _empty_result(question: str, narrative: str, reason: str) -> EVARResult:
    return EVARResult(
        claims=(),
        budget_max_rounds=0,
        budget_max_hypotheses=0,
        n_uncertain=0,
        rounds=(),
        admitted=(),
        quarantined=(),
        discarded=(),
        answer="",
        stop_reason=reason,
        covered_slots=(),
        unresolved_slots=(),
        question=question,
        narrative=narrative,
    )


def _synthesize(admitted: Sequence[Hypothesis], question: str) -> str:
    """One paragraph grounded ONLY in admitted hypotheses."""
    if not admitted:
        return (
            "No hypothesis survived evidence-validated admission, so the "
            "answer-supporting state is empty. The question remains unresolved "
            f"on the locked store alone: {question.strip() or '(no question)'}"
        )
    bits: list[str] = []
    for h in admitted:
        s = h.statement.strip()
        if s and not s.endswith("."):
            s += "."
        if s and s not in bits:
            bits.append(s)
    body = " ".join(bits)
    return (
        "Grounded only in admitted hypotheses (quarantined rumor and discarded "
        f"contradictions are excluded): {body}"
    )


class EVAREngine:
    """Budget-aware test-time loop with verifier-gated hypothesis admission."""

    def __init__(
        self,
        *,
        reasoner: Reasoner | None = None,
        max_rounds: int | None = None,
        max_hypotheses_per_round: int = 3,
    ) -> None:
        self.reasoner: Reasoner = reasoner or build_reasoner()
        self.max_rounds_override = max_rounds
        self.max_hypotheses_per_round = max_hypotheses_per_round

    def run(self, narrative: str, question: str) -> EVARResult:
        nar = narrative if isinstance(narrative, str) else ""
        ques = question if isinstance(question, str) else ""
        if not nar.strip() or not ques.strip():
            return _empty_result(ques, nar, StopReason.EMPTY_INPUT.value)

        claims = compile_narrative(nar)
        store = EvidenceStore()
        store.extend(claims)
        store.lock()

        gaps = parse_slots(ques)
        n_uncertain = sum(
            1 for c in claims if c.status.value != "OK" or c.is_rumor
        )
        unresolved = unresolved_gaps(gaps, admitted=())
        budget = Budget.from_signals(
            n_unresolved_gaps=len(unresolved),
            n_uncertain=n_uncertain,
            max_rounds=self.max_rounds_override,
            max_hypotheses_per_round=self.max_hypotheses_per_round,
        )

        admitted: list[Hypothesis] = []
        quarantined: list[Hypothesis] = []
        discarded: list[Hypothesis] = []
        already_tried: list[str] = []
        rounds: list[RoundLog] = []
        stop = StopReason.BUDGET_EXHAUSTED.value

        if budget.max_rounds == 0:
            return EVARResult(
                claims=tuple(claims),
                budget_max_rounds=0,
                budget_max_hypotheses=budget.max_hypotheses,
                n_uncertain=n_uncertain,
                rounds=(),
                admitted=(),
                quarantined=(),
                discarded=(),
                answer=_synthesize((), ques),
                stop_reason=StopReason.FAST_PATH.value,
                covered_slots=covered_slots(gaps, admitted),
                unresolved_slots=tuple(g.slot for g in unresolved_gaps(gaps, admitted)),
                question=ques,
                narrative=nar,
            )

        for t in range(budget.max_rounds):
            if not budget.can_continue():
                stop = StopReason.BUDGET_EXHAUSTED.value
                break
            open_gaps = unresolved_gaps(gaps, admitted)
            if not open_gaps:
                stop = StopReason.SUFFICIENCY.value
                break

            proposed = list(
                self.reasoner.propose(open_gaps, store, ques, already_tried)
            )[: budget.max_hypotheses_per_round]
            # Honor remaining hypothesis tokens.
            room = budget.remaining_hypotheses()
            proposed = proposed[:room]
            if not proposed:
                stop = StopReason.NO_CANDIDATES.value
                break

            budget.consume_hypotheses(len(proposed))
            challenges = []
            records = []
            for hyp in proposed:
                already_tried.append(hyp.statement)
                chal = construct_challenge(hyp, store)
                rec = validate(hyp, store, challenge=chal)
                challenges.append(chal)
                records.append(rec)
                if rec.verdict == Verdict.ADMITTED:
                    admitted.append(hyp)
                elif rec.verdict == Verdict.DISCARDED:
                    discarded.append(hyp)
                else:
                    quarantined.append(hyp)

            budget.consume_round()
            open_after = unresolved_gaps(gaps, admitted)
            rounds.append(
                RoundLog(
                    round_index=t + 1,
                    proposed=tuple(proposed),
                    challenges=tuple(challenges),
                    verdicts=tuple(records),
                    remaining_rounds=budget.remaining_rounds(),
                    remaining_hypotheses=budget.remaining_hypotheses(),
                    unresolved_slots=tuple(g.slot for g in open_after),
                    notes=f"admitted={len(admitted)} quarantined={len(quarantined)} discarded={len(discarded)}",
                )
            )

            if is_sufficient(gaps, admitted):
                stop = StopReason.SUFFICIENCY.value
                break
            if not budget.can_continue():
                stop = StopReason.BUDGET_EXHAUSTED.value
                break
        else:
            # loop finished without break
            if is_sufficient(gaps, admitted):
                stop = StopReason.SUFFICIENCY.value
            else:
                stop = StopReason.BUDGET_EXHAUSTED.value

        return EVARResult(
            claims=tuple(claims),
            budget_max_rounds=budget.max_rounds,
            budget_max_hypotheses=budget.max_hypotheses,
            n_uncertain=n_uncertain,
            rounds=tuple(rounds),
            admitted=tuple(admitted),
            quarantined=tuple(quarantined),
            discarded=tuple(discarded),
            answer=_synthesize(admitted, ques),
            stop_reason=stop,
            covered_slots=covered_slots(gaps, admitted),
            unresolved_slots=tuple(g.slot for g in unresolved_gaps(gaps, admitted)),
            question=ques,
            narrative=nar,
        )


def run_evar(
    narrative: str,
    question: str,
    *,
    reasoner: Reasoner | None = None,
    max_rounds: int | None = None,
) -> EVARResult:
    """Public convenience wrapper around ``EVAREngine.run``."""
    settings = EVARSettings.from_env()
    r = reasoner or build_reasoner(settings.reasoner)
    return EVAREngine(reasoner=r, max_rounds=max_rounds).run(narrative, question)
