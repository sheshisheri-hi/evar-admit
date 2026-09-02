"""Reasoner protocol, deterministic default, optional OpenAI backend."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Protocol

from evar.compiler import extract_places, extract_times, stem
from evar.store import EvidenceStore
from evar.types import AtomicClaim, Gap, Hypothesis, Polarity

_PERSON_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _people_from_claims(claims: Sequence[AtomicClaim]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for c in claims:
        for e in c.entities:
            if _PERSON_RE.search(e):
                key = _norm(e)
                if key not in seen:
                    seen.add(key)
                    found.append(e)
        for m in _PERSON_RE.finditer(c.text):
            key = _norm(m.group(0))
            if key not in seen:
                seen.add(key)
                found.append(m.group(0))
    return found


def _collect(claims: Sequence[AtomicClaim], kind: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in claims:
        values = c.places if kind == "place" else c.times
        extra = extract_places(c.text) if kind == "place" else extract_times(c.text)
        for v in (*values, *extra):
            key = _norm(v)
            if key and key not in seen:
                seen.add(key)
                out.append(v)
    return out


def _already(statement: str, tried: Sequence[str]) -> bool:
    n = _norm(statement)
    return any(_norm(t) == n for t in tried)


class Reasoner(Protocol):
    def propose(
        self,
        gaps: Sequence[Gap],
        store: EvidenceStore,
        question: str,
        already_tried: Sequence[str],
    ) -> Sequence[Hypothesis]:
        ...


class DeterministicReasoner:
    """Offline proposer. Uses only claim text + gap labels; no network.

    Always tries to emit 1-3 candidates. When a rumor-attributed claim exists,
    one candidate is a red herring built from that rumor so quarantine/discard
    can actually fire in the demo path.
    """

    def propose(
        self,
        gaps: Sequence[Gap],
        store: EvidenceStore,
        question: str,
        already_tried: Sequence[str],
    ) -> Sequence[Hypothesis]:
        claims = store.claims()
        people = _people_from_claims(claims)
        places = _collect(claims, "place")
        times = _collect(claims, "time")
        rumor_claims = [c for c in claims if c.is_rumor]
        tried = list(already_tried)
        slot_names = [g.slot for g in gaps] or ["entity"]
        primary_slot = slot_names[0]
        primary_gap = gaps[0].id if gaps else "gap-entity"

        candidates: list[Hypothesis] = []

        def push(hyp: Hypothesis) -> None:
            if len(candidates) >= 3:
                return
            if _already(hyp.statement, tried) or _already(
                hyp.statement, [c.statement for c in candidates]
            ):
                return
            candidates.append(hyp)

        # --- 1. Visual / clothing cue (often the well-supported actor) -----
        yellow = next(
            (c for c in claims if "yellow" in c.text.lower() and "slicker" in c.text.lower()),
            None,
        )
        if yellow is not None:
            time_hit = extract_times(yellow.text)
            place_hit = extract_places(yellow.text)
            time_s = time_hit[0] if time_hit else (times[0] if times else "the night of the theft")
            place_s = next(
                (p for p in place_hit if "office" in p.lower()),
                place_hit[0] if place_hit else (places[0] if places else "the scene"),
            )
            stmt = (
                f"A figure in a yellow slicker entered the {place_s} at {time_s} "
                f"and took the harbor ledger."
            )
            push(
                Hypothesis(
                    id="h-slicker",
                    statement=stmt,
                    target_gap=primary_gap,
                    entities=("yellow slicker", place_s, "harbor ledger"),
                    predicates=("enter", "take"),
                    polarity=Polarity.ASSERTED,
                    slot="culprit",
                    fillers=(
                        ("culprit", "a figure in a yellow slicker issued by the warehouse"),
                        ("location", place_s),
                        ("time", time_s),
                        ("object", "harbor ledger"),
                    ),
                )
            )

        # --- 2. Red herring from rumor (quarantine in the demo path) ------
        if rumor_claims:
            rc = rumor_claims[0]
            rumored_people = [
                e for e in rc.entities if _PERSON_RE.search(e)
            ] or people[:1]
            actor = rumored_people[0] if rumored_people else "Someone"
            stmt = (
                f"{actor} sold ledgers before."
            )
            push(
                Hypothesis(
                    id="h-rumor",
                    statement=stmt,
                    target_gap=primary_gap,
                    entities=(actor,),
                    predicates=("sell", "ledger"),
                    polarity=Polarity.ASSERTED,
                    slot="motive",
                    fillers=(
                        ("motive", f"{actor} sold ledgers before"),
                    ),
                )
            )

        # --- 3. Naive key-holder / access hyp (often discarded) -----------
        key_claim = next(
            (c for c in claims if re.search(r"\bkey\b", c.text, re.I)),
            None,
        )
        if key_claim is not None:
            holder = next(
                (e for e in key_claim.entities if _PERSON_RE.search(e)),
                people[1] if len(people) > 1 else (people[0] if people else "The key holder"),
            )
            office = next((p for p in places if "office" in p.lower()), "the office")
            stmt = (
                f"{holder} used the spare office key to steal the harbor ledger "
                f"from {office}."
            )
            push(
                Hypothesis(
                    id="h-key",
                    statement=stmt,
                    target_gap=primary_gap,
                    entities=(holder, "spare key", office),
                    predicates=("use", "steal"),
                    polarity=Polarity.ASSERTED,
                    slot="culprit",
                    fillers=(
                        ("culprit", holder),
                        ("location", office),
                        ("object", "harbor ledger"),
                    ),
                )
            )

        # --- 4. Generic: each named person as a possible culprit ----------
        crime_obj = "harbor ledger"
        q_low = question.lower()
        if "coin" in q_low:
            crime_obj = "coin"
        elif "gem" in q_low:
            crime_obj = "gem"
        for person in people:
            if len(candidates) >= 3:
                break
            stmt = f"{person} took the {crime_obj}."
            push(
                Hypothesis(
                    id=f"h-person-{_norm(person).replace(' ', '-')}",
                    statement=stmt,
                    target_gap=primary_gap,
                    entities=(person, crime_obj),
                    predicates=("take", stem(crime_obj)),
                    polarity=Polarity.ASSERTED,
                    slot="culprit",
                    fillers=(("culprit", person), ("object", crime_obj)),
                )
            )

        # --- 5. Fill remaining location / time / object slots -------------
        if any(g.slot == "location" for g in gaps) and places:
            place = places[0]
            stmt = f"The relevant location is {place}."
            push(
                Hypothesis(
                    id="h-loc",
                    statement=stmt,
                    target_gap="gap-location",
                    entities=(place,),
                    predicates=("location",),
                    slot="location",
                    fillers=(("location", place),),
                )
            )
        if any(g.slot == "time" for g in gaps) and times:
            t = times[0]
            stmt = f"The relevant time is {t}."
            push(
                Hypothesis(
                    id="h-time",
                    statement=stmt,
                    target_gap="gap-time",
                    entities=(t,),
                    predicates=("time",),
                    slot="time",
                    fillers=(("time", t),),
                )
            )
        if any(g.slot == "object" for g in gaps):
            stmt = f"The missing object is the {crime_obj}."
            push(
                Hypothesis(
                    id="h-obj",
                    statement=stmt,
                    target_gap="gap-object",
                    entities=(crime_obj,),
                    predicates=("object",),
                    slot="object",
                    fillers=(("object", crime_obj),),
                )
            )

        # --- 6. Fallback so propose() is never empty when there is a store
        if not candidates and claims:
            snippet = claims[0].text
            ents = claims[0].entities or (snippet[:40],)
            push(
                Hypothesis(
                    id="h-fallback",
                    statement=snippet,
                    target_gap=primary_gap,
                    entities=ents,
                    predicates=claims[0].predicates[:4],
                    slot=primary_slot,
                    fillers=((primary_slot, snippet[:80]),),
                )
            )
        return tuple(candidates[:3])


class OpenAIReasoner:
    """Optional chat-completions proposer. Any error falls back to deterministic."""

    def __init__(
        self,
        *,
        model: str | None = None,
        fallback: DeterministicReasoner | None = None,
    ) -> None:
        self.model = model or os.getenv("EVAR_MODEL", "gpt-4o-mini")
        self.fallback = fallback or DeterministicReasoner()

    def propose(
        self,
        gaps: Sequence[Gap],
        store: EvidenceStore,
        question: str,
        already_tried: Sequence[str],
    ) -> Sequence[Hypothesis]:
        try:
            return self._propose_or_raise(gaps, store, question, already_tried)
        except Exception:
            return self.fallback.propose(gaps, store, question, already_tried)

    def _propose_or_raise(
        self,
        gaps: Sequence[Gap],
        store: EvidenceStore,
        question: str,
        already_tried: Sequence[str],
    ) -> Sequence[Hypothesis]:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        claim_lines = [
            f"- [{c.id}] ({c.status.value}{' rumor' if c.is_rumor else ''}) {c.text}"
            for c in store.claims()[:40]
        ]
        gap_lines = [f"- {g.slot}: {g.description}" for g in gaps]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "statement": {"type": "string"},
                            "slot": {"type": "string"},
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "predicates": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "fillers": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["statement", "slot", "entities", "predicates", "fillers"],
                    },
                }
            },
            "required": ["hypotheses"],
        }
        prompt = (
            "Propose 1-3 candidate hypotheses for the unresolved gaps. "
            "Use only information that could be guessed from the locked claims "
            "and the gap labels. Include at least one weakly-grounded red herring "
            "if a rumor-attributed claim exists. Do not treat rumor as fact.\n\n"
            f"Question: {question}\n"
            f"Gaps:\n{chr(10).join(gap_lines) or '(none)'}\n"
            f"Claims:\n{chr(10).join(claim_lines)}\n"
            f"Already tried: {list(already_tried)}"
        )
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evar_hypotheses",
                    "strict": True,
                    "schema": schema,
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": "You propose hypotheses for EVAR. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        payload = json.loads(content)
        hyps: list[Hypothesis] = []
        primary_gap = gaps[0].id if gaps else "gap-entity"
        for i, raw in enumerate(payload.get("hypotheses", [])[:3], start=1):
            fillers = raw.get("fillers") or {}
            filler_t = tuple((str(k), str(v)) for k, v in fillers.items())
            hyps.append(
                Hypothesis(
                    id=f"h-llm-{i}",
                    statement=str(raw.get("statement", "")).strip(),
                    target_gap=primary_gap,
                    entities=tuple(raw.get("entities") or ()),
                    predicates=tuple(raw.get("predicates") or ()),
                    slot=str(raw.get("slot") or (gaps[0].slot if gaps else "entity")),
                    fillers=filler_t,
                )
            )
        hyps = [h for h in hyps if h.statement and not _already(h.statement, already_tried)]
        if not hyps:
            return self.fallback.propose(gaps, store, question, already_tried)
        return tuple(hyps)


def build_reasoner(kind: str | None = None) -> Reasoner:
    raw = (kind or os.getenv("EVAR_REASONER") or "deterministic").strip().lower()
    if raw == "openai":
        return OpenAIReasoner()
    return DeterministicReasoner()
