"""Edge-case tests for the EVAR admission loop."""

from __future__ import annotations

import pytest

from evar.compiler import compile_narrative
from evar.engine import EVAREngine, run_evar
from evar.store import EvidenceStore
from evar.types import (
    FrozenStoreError,
    Gap,
    Hypothesis,
    Polarity,
    Verdict,
)
from evar.validator import validate


def _store_of(narrative: str) -> EvidenceStore:
    store = EvidenceStore()
    store.extend(compile_narrative(narrative))
    store.lock()
    return store


def _hyp(
    statement: str,
    *,
    entities: tuple[str, ...] = (),
    predicates: tuple[str, ...] = (),
    slot: str | None = "culprit",
    fillers: tuple[tuple[str, str], ...] = (),
    hyp_id: str = "h-test",
) -> Hypothesis:
    return Hypothesis(
        id=hyp_id,
        statement=statement,
        target_gap="gap-culprit",
        entities=entities,
        predicates=predicates,
        polarity=Polarity.ASSERTED,
        slot=slot,
        fillers=fillers,
    )


# ---------------------------------------------------------------------------
# 1. compiler source spans
# ---------------------------------------------------------------------------
def test_compiler_source_spans_slice_back_to_narrative() -> None:
    narrative = (
        "Alice stole the coin from the kitchen at noon. "
        "Bob was never in the kitchen that day."
    )
    claims = compile_narrative(narrative)
    assert claims, "compiler dropped the whole narrative"
    for claim in claims:
        sliced = narrative[claim.source.start : claim.source.end]
        assert sliced == claim.source.text, (claim.id, sliced, claim.source.text)
        assert claim.source.text in narrative
        assert 0 <= claim.source.start < claim.source.end <= len(narrative)
    # every original sentence produced at least one claim
    assert any("Alice stole" in c.text for c in claims)
    assert any("Bob" in c.text for c in claims)


# ---------------------------------------------------------------------------
# 2. lock then mutate
# ---------------------------------------------------------------------------
def test_lock_then_mutate_raises_frozen_store_error() -> None:
    store = EvidenceStore()
    store.extend(compile_narrative("Alice waited in the park."))
    store.lock()
    assert store.locked
    extra = compile_narrative("Secret extra sentence.")[0]
    extra = extra.__class__(
        **{**extra.__dict__, "id": "c999"}
    )
    with pytest.raises(FrozenStoreError):
        store.add(extra)
    with pytest.raises(FrozenStoreError):
        store.remove(store.claims()[0].id)


# ---------------------------------------------------------------------------
# 3. supported hypothesis → ADMITTED
# ---------------------------------------------------------------------------
def test_supported_hypothesis_is_admitted() -> None:
    store = _store_of("Alice stole the coin from the kitchen at noon.")
    hyp = _hyp(
        "Alice stole the coin from the kitchen.",
        entities=("Alice", "coin", "kitchen"),
        predicates=("steal",),
        fillers=(("culprit", "Alice"), ("object", "coin"), ("location", "kitchen")),
    )
    rec = validate(hyp, store)
    assert rec.verdict == Verdict.ADMITTED
    assert rec.support_score >= 0.5
    assert rec.contradicted is False


# ---------------------------------------------------------------------------
# 4. contradiction → DISCARDED
# ---------------------------------------------------------------------------
def test_contradictory_hypothesis_is_discarded() -> None:
    store = _store_of(
        "Bob was not in the kitchen. "
        "The coin was stolen from the kitchen at noon. "
        "Bob was in the warehouse until 11pm."
    )
    hyp = _hyp(
        "Bob stole the coin from the kitchen.",
        entities=("Bob", "coin", "kitchen"),
        predicates=("steal",),
        fillers=(("culprit", "Bob"), ("location", "kitchen"), ("object", "coin")),
    )
    rec = validate(hyp, store)
    assert rec.verdict == Verdict.DISCARDED
    assert rec.contradicted is True


# ---------------------------------------------------------------------------
# 5. rumor / unverifiable → QUARANTINED
# ---------------------------------------------------------------------------
def test_rumor_hypothesis_is_quarantined() -> None:
    store = _store_of(
        "Alice was in the park at noon. "
        "A rumor circulating on the dock claimed that Alice sold coins before."
    )
    hyp = _hyp(
        "Alice stole the coin because a rumor says Alice sold coins before.",
        entities=("Alice",),
        predicates=("steal", "sell", "coin"),
        fillers=(("culprit", "Alice"), ("motive", "sold coins before (rumor)")),
    )
    rec = validate(hyp, store)
    assert rec.verdict == Verdict.QUARANTINED
    assert rec.support_score < 0.5
    assert rec.contradicted is False


# ---------------------------------------------------------------------------
# 6. budget exhausted stops even if gaps remain
# ---------------------------------------------------------------------------
class _UselessReasoner:
    def propose(self, gaps, store, question, already_tried):
        n = len(list(already_tried)) + 1
        return [
            _hyp(
                f"The moon is made of green cheese variant {n}.",
                entities=("moon",),
                predicates=("cheese",),
                slot="motive",
                fillers=(("motive", "cheese"),),
                hyp_id=f"h-moon-{n}",
            )
        ]


def test_budget_exhausted_stops_even_if_gaps_remain() -> None:
    narrative = (
        "The harbor ledger was stolen Tuesday night from the customs office. "
        "Alice clocked out at 9:10pm."
    )
    question = "Who stole the ledger, where, when, and why?"
    engine = EVAREngine(reasoner=_UselessReasoner(), max_rounds=2)
    result = engine.run(narrative, question)
    assert result.stop_reason == "budget_exhausted"
    assert len(result.rounds) == 2
    assert result.unresolved_slots  # gaps remain
    assert result.budget_max_rounds == 2


# ---------------------------------------------------------------------------
# 7. sufficiency stop: no extra rounds once slots are covered
# ---------------------------------------------------------------------------
class _CountingCoverReasoner:
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, gaps, store, question, already_tried):
        self.calls += 1
        if self.calls == 1:
            return [
                _hyp(
                    "Alice stole the coin from the kitchen.",
                    entities=("Alice", "coin", "kitchen"),
                    predicates=("steal",),
                    fillers=(("culprit", "Alice"), ("object", "coin")),
                    hyp_id="h-cover",
                )
            ]
        return [
            _hyp(
                "This extra round should never run.",
                entities=("Nobody",),
                predicates=("run",),
                hyp_id="h-extra",
            )
        ]


def test_sufficiency_stops_without_extra_rounds() -> None:
    reasoner = _CountingCoverReasoner()
    engine = EVAREngine(reasoner=reasoner, max_rounds=6)
    result = engine.run(
        "Alice stole the coin from the kitchen at noon.",
        "Who took the coin?",
    )
    assert result.stop_reason == "sufficiency"
    assert len(result.rounds) == 1
    assert reasoner.calls == 1
    assert "culprit" in result.covered_slots
    assert result.admitted


# ---------------------------------------------------------------------------
# 8. empty narrative / empty question: no traceback
# ---------------------------------------------------------------------------
def test_empty_narrative_or_question_is_clean() -> None:
    r1 = run_evar("", "Who did it?")
    r2 = run_evar("Alice waited in the park.", "")
    r3 = run_evar("   ", "   ")
    for r in (r1, r2, r3):
        assert r.stop_reason == "empty_input"
        assert r.answer == ""
        assert r.rounds == ()
        assert r.admitted == ()


# ---------------------------------------------------------------------------
# 9. admitted answer must not include quarantined rumor text
# ---------------------------------------------------------------------------
def test_admitted_answer_excludes_quarantined_rumor_text() -> None:
    narrative = (
        "The harbor ledger was stolen Tuesday night from the customs office. "
        "A customs drone frame time-stamped 10:22pm shows a figure in a yellow "
        "slicker entering the customs office. The warehouse issues yellow slickers. "
        "A rumor circulating among the night crew claimed that Mara Cole had "
        "sold ledgers before. Mara Cole clocked out at 9:10pm. Mara Cole was "
        "seen on the south ferry at 9:40pm."
    )
    question = "Who stole the harbor ledger, and what evidence actually supports that?"
    result = run_evar(narrative, question)
    rumor_snippets = (
        "sold ledgers before",
        "sold ledgers",
        "rumor says",
        "a rumor",
    )
    answer_l = result.answer.lower()
    for snippet in rumor_snippets:
        assert snippet not in answer_l, (snippet, result.answer)
    quarantined_text = " ".join(h.statement.lower() for h in result.quarantined)
    # the rumor should have been proposed and quarantined
    assert "sold" in quarantined_text or any(
        "rumor" in h.statement.lower() for h in result.quarantined
    )


# ---------------------------------------------------------------------------
# 10. validator uses only the locked store, not previously admitted hyps
# ---------------------------------------------------------------------------
def test_validator_ignores_previously_admitted_hypotheses() -> None:
    store = _store_of("Alice was in the park at noon.")
    # A fluent but unsupported statement that looks like it was "already admitted"
    false_admitted = _hyp(
        "Alice stole the gems from the vault.",
        entities=("Alice", "gems", "vault"),
        predicates=("steal",),
        fillers=(("culprit", "Alice"), ("object", "gems")),
        hyp_id="h-false-admitted",
    )
    # The candidate to validate is the same unsupported content.
    rec = validate(false_admitted, store)
    assert rec.verdict != Verdict.ADMITTED
    assert rec.verdict == Verdict.QUARANTINED
    # Sanity: the store really does not contain the planted statement.
    joined = " ".join(c.text.lower() for c in store.claims())
    assert "gems" not in joined
    assert "vault" not in joined
    # And validate() does not even accept an admitted-hypotheses argument.
    assert validate.__code__.co_varnames[0:3] == ("hypothesis", "store", "challenge") or (
        "admitted" not in validate.__code__.co_varnames
    )


def test_parse_slots_always_has_at_least_one() -> None:
    from evar.gaps import parse_slots

    gaps = parse_slots("Explain the incident.")
    assert gaps and gaps[0].slot in {"entity", "culprit", "object", "location", "time", "motive"}
