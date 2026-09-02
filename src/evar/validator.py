"""Hypothesis-conditioned validation challenges against the locked store.

Verification never consults previously admitted hypotheses. Support is
attested only by locked, non-rumor claims. That is the paper's admission
boundary: a fluent but unsupported intermediate cannot become evidence
for the next round by having been generated in this one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from evar.compiler import extract_places, extract_times, stem
from evar.store import EvidenceStore
from evar.types import (
    AtomicClaim,
    Challenge,
    ClaimStatus,
    Hypothesis,
    Polarity,
    Verdict,
    VerdictRecord,
)

SUPPORT_THRESHOLD = 0.5

_PERSON_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_NON_PERSON = {
    "the", "a", "an", "on", "in", "at", "to", "of", "for", "and", "or",
    "wednesday", "tuesday", "monday", "thursday", "friday", "saturday", "sunday",
    "office", "harbor", "harbour", "ferry", "warehouse", "locker", "pier",
    "dock", "drawer", "counter", "bay", "desk", "camera", "kitchen", "park",
    "street", "yellow", "slicker", "figure", "ledger", "book", "key", "rain",
    "night", "morning", "customs", "south", "north", "loading", "spare",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    return {stem(t) for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2}


def _looks_person(name: str) -> bool:
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return False
    if any(p.lower() in _NON_PERSON or p.lower() in {"office", "kitchen", "park"} for p in parts):
        return False
    if not parts[0][:1].isupper():
        return False
    if parts[0].lower() in _NON_PERSON:
        return False
    return True


def _people(items: Sequence[str], extra_text: str = "") -> set[str]:
    found: set[str] = set()
    for e in items:
        if _looks_person(e):
            found.add(_norm(e))
    for m in _PERSON_RE.finditer(extra_text):
        if _looks_person(m.group(0)):
            found.add(_norm(m.group(0)))
    return found


def _usable(claim: AtomicClaim) -> bool:
    """Rumor / uncertain claims are not allowed to attest support."""
    if claim.is_rumor:
        return False
    if claim.status == ClaimStatus.UNCERTAIN:
        return False
    return True


def construct_challenge(hypothesis: Hypothesis, store: EvidenceStore) -> Challenge:
    """Build support / contradiction / prerequisite queries for H against B."""
    support_q: list[str] = []
    for ent in hypothesis.entities:
        if ent.strip():
            support_q.append(ent.strip())
    for pred in hypothesis.predicates:
        if pred.strip() and pred.lower() not in {e.lower() for e in hypothesis.entities}:
            support_q.append(pred.strip())
    if not support_q:
        support_q = [w for w in _tokens(hypothesis.statement)]

    people = _people(hypothesis.entities, hypothesis.statement)
    places = {_norm(p) for p in extract_places(hypothesis.statement)}
    times = {_norm(t) for t in extract_times(hypothesis.statement)}

    contradiction_q: list[str] = []
    if people:
        contradiction_q.append(
            "negated claim about " + ", ".join(sorted(people))
        )
    if people and places:
        contradiction_q.append(
            "exclusive location for "
            + ", ".join(sorted(people))
            + " other than "
            + ", ".join(sorted(places))
        )
    if people and times:
        contradiction_q.append(
            "exclusive time for "
            + ", ".join(sorted(people))
            + " other than "
            + ", ".join(sorted(times))
        )
    if hypothesis.polarity == Polarity.ASSERTED:
        contradiction_q.append("opposite polarity on overlapping entities")

    # Prerequisite: if H asserts a theft/action, the object of the crime
    # should exist as a locked claim.
    prereq: list[str] = []
    fillers = hypothesis.filler_map()
    if "object" in fillers:
        prereq.append(f"object mentioned in store: {fillers['object']}")
    if any(stem(p) in {"steal", "enter", "use"} for p in hypothesis.predicates):
        prereq.append("action-relevant locked claim exists")

    return Challenge(
        hypothesis_id=hypothesis.id,
        support_queries=tuple(dict.fromkeys(support_q)),
        contradiction_queries=tuple(dict.fromkeys(contradiction_q)),
        prerequisite_queries=tuple(prereq),
    )


def _claim_contains(claim: AtomicClaim, query: str) -> bool:
    q = _norm(query)
    q_stem = stem(q.replace(" ", ""))
    hay = _norm(claim.text)
    hay_tokens = _tokens(claim.text) | _tokens(" ".join(claim.entities)) | set(
        stem(p) for p in claim.predicates
    )
    if q in hay or q in _norm(" ".join(claim.entities)):
        return True
    if stem(q) in hay_tokens or q_stem in hay_tokens:
        return True
    # multi-word entity: all tokens present
    parts = [stem(p) for p in q.split() if len(p) > 2]
    if parts and all(p in hay_tokens for p in parts):
        return True
    return False


def _actor_aligned_claims(
    hypothesis: Hypothesis, claims: Sequence[AtomicClaim]
) -> list[AtomicClaim]:
    people = _people(hypothesis.entities, hypothesis.statement)
    if not people:
        # no person actor: overlap on any entity or place
        h_ents = {_norm(e) for e in hypothesis.entities} | _tokens(hypothesis.statement)
        aligned: list[AtomicClaim] = []
        for c in claims:
            c_ents = {_norm(e) for e in c.entities} | _tokens(c.text)
            if h_ents & c_ents:
                aligned.append(c)
        return aligned
    aligned = []
    for c in claims:
        c_people = _people(c.entities, c.text)
        if people & c_people:
            aligned.append(c)
    return aligned


def support_score(
    hypothesis: Hypothesis,
    store: EvidenceStore,
    challenge: Challenge | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Fraction of H's support queries attested by locked, usable claims.

    Predicates must be attested by actor-aligned claims (sharing a person
    entity when H names one). Bare entity mentions may be attested by any
    usable claim. Rumor-tagged units never count.
    """
    queries = list(challenge.support_queries) if challenge else list(hypothesis.entities) + list(
        hypothesis.predicates
    )
    queries = [q for q in queries if q.strip()]
    if not queries:
        return 0.0, ()

    usable = [c for c in store.claims() if _usable(c)]
    aligned = _actor_aligned_claims(hypothesis, usable)
    h_people = _people(hypothesis.entities, hypothesis.statement)
    supporting_ids: list[str] = []
    attested = 0

    for q in queries:
        is_entity = any(_norm(q) == _norm(e) for e in hypothesis.entities)
        pool = usable if is_entity else (aligned if h_people else usable)
        # predicates (non-entity queries) require overlapping entities
        hit = False
        for claim in pool:
            if hypothesis.polarity == Polarity.ASSERTED and claim.polarity == Polarity.NEGATED:
                # a negated claim does not assert the predicate
                if not is_entity:
                    continue
            if _claim_contains(claim, q):
                hit = True
                if claim.id not in supporting_ids:
                    supporting_ids.append(claim.id)
        if hit:
            attested += 1
    return attested / len(queries), tuple(supporting_ids)


def find_contradictions(
    hypothesis: Hypothesis, store: EvidenceStore
) -> tuple[bool, tuple[str, ...]]:
    """Contradiction: overlapping entities + opposite polarity, or exclusive slot fillers."""
    h_people = _people(hypothesis.entities, hypothesis.statement)
    h_places = {_norm(p) for p in extract_places(hypothesis.statement)}
    for _slot, val in hypothesis.fillers:
        if _slot == "location":
            h_places.add(_norm(val))
        extra_places = extract_places(val)
        h_places.update(_norm(p) for p in extra_places)
    h_times = {_norm(t) for t in extract_times(hypothesis.statement)}
    h_tokens = _tokens(hypothesis.statement)
    h_ents = {_norm(e) for e in hypothesis.entities}

    ids: list[str] = []
    for claim in store.claims():
        if claim.is_rumor:
            continue
        c_people = _people(claim.entities, claim.text)
        c_places = {_norm(p) for p in claim.places} | {_norm(p) for p in extract_places(claim.text)}
        c_ents = {_norm(e) for e in claim.entities}
        shared_people = h_people & c_people
        shared_ents = (h_ents & c_ents) or shared_people
        if not shared_ents and not shared_people:
            continue

        # opposite polarity on overlapping entities / related predicates
        if shared_people and claim.polarity != hypothesis.polarity:
            c_toks = _tokens(claim.text)
            # related if they share a content stem besides the person name
            person_stems: set[str] = set()
            for p in shared_people:
                person_stems.update(_tokens(p))
            overlap = (h_tokens & c_toks) - person_stems - {"the", "and"}
            place_overlap = bool(h_places & c_places)
            # A negated "was not seen near the office" vs asserted "was at the office"
            if overlap or place_overlap or (h_places and c_places):
                ids.append(claim.id)
                continue

        # exclusive locations for the same actor
        if shared_people and h_places and c_places and h_places.isdisjoint(c_places):
            # closed-world: two different places for the same person is a clash
            ids.append(claim.id)
            continue

        # exclusive times for the same actor (different clock times)
        c_times = {_norm(t) for t in claim.times} | {_norm(t) for t in extract_times(claim.text)}
        clock = {t for t in h_times if re.search(r"\d", t)}
        c_clock = {t for t in c_times if re.search(r"\d", t)}
        if shared_people and clock and c_clock and clock.isdisjoint(c_clock):
            # only contradict if places also disagree or hyp asserts presence at crime scene
            if h_places and c_places and h_places.isdisjoint(c_places):
                ids.append(claim.id)

    return (len(ids) > 0), tuple(dict.fromkeys(ids))


def validate(
    hypothesis: Hypothesis,
    store: EvidenceStore,
    *,
    challenge: Challenge | None = None,
) -> VerdictRecord:
    """Score H against the locked store only. Admitted hyps are invisible here.

    Thresholds:
      support >= 0.5 and no contradiction -> ADMITTED
      contradiction                         -> DISCARDED
      otherwise                             -> QUARANTINED
    """
    chal = challenge or construct_challenge(hypothesis, store)
    score, support_ids = support_score(hypothesis, store, chal)
    contradicted, contra_ids = find_contradictions(hypothesis, store)

    if contradicted:
        verdict = Verdict.DISCARDED
        note = "contradicts locked evidence"
    elif score >= SUPPORT_THRESHOLD:
        verdict = Verdict.ADMITTED
        note = "supported by locked evidence"
    else:
        verdict = Verdict.QUARANTINED
        note = "unverifiable against locked evidence"

    return VerdictRecord(
        hypothesis_id=hypothesis.id,
        verdict=verdict,
        support_score=round(score, 4),
        contradicted=contradicted,
        supporting_claim_ids=support_ids,
        contradicting_claim_ids=contra_ids,
        note=note,
    )
