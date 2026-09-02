"""Required answer slots from the question; unresolved vs covered by admitted hyps."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from evar.compiler import extract_names, extract_places, extract_times
from evar.types import Gap, Hypothesis

_SLOT_CUES: dict[str, tuple[str, ...]] = {
    "culprit": (
        "who", "whom", "culprit", "perpetrator", "thief", "suspect",
        "stole", "stolen", "killer", "attacker", "guilty",
    ),
    "location": ("where", "location", "place", "scene"),
    "time": ("when", "what time", "what night", "what day"),
    "motive": ("why", "motive", "reason", "intent"),
    "object": ("what", "which object", "which item", "ledger", "object", "item"),
}

_PERSON_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")


def parse_slots(question: str) -> list[Gap]:
    """Parse the question into required slots.

    Always includes at least one slot: a matched cue, otherwise a main-entity
    slot derived from the question's primary noun phrase.
    """
    q = (question or "").strip()
    if not q:
        return []

    low = q.lower()
    gaps: list[Gap] = []
    used: set[str] = set()

    for slot, cues in _SLOT_CUES.items():
        if any(re.search(rf"\b{re.escape(cue)}\b", low) for cue in cues):
            if slot not in used:
                used.add(slot)
                gaps.append(
                    Gap(
                        id=f"gap-{slot}",
                        slot=slot,
                        description=f"unresolved {slot} required by the question",
                        required=True,
                    )
                )

    if not gaps:
        names = extract_names(q)
        main = names[0] if names else _main_noun(q)
        gaps.append(
            Gap(
                id="gap-entity",
                slot="entity",
                description=f"main entity slot ({main})" if main else "main entity slot",
                required=True,
            )
        )
    return gaps


def _main_noun(question: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", question)
    words = [w for w in cleaned.split() if len(w) > 3]
    return words[0] if words else "entity"


def _hyp_times(h: Hypothesis) -> set[str]:
    found = {t.lower() for t in extract_times(h.statement)}
    for _slot, val in h.fillers:
        found.update(t.lower() for t in extract_times(val))
    return found


def _hyp_places(h: Hypothesis) -> set[str]:
    found = {p.lower() for p in extract_places(h.statement)}
    for _slot, val in h.fillers:
        found.update(p.lower() for p in extract_places(val))
        if _slot == "location" and val.strip():
            found.add(val.lower())
    return found


def _hyp_people(h: Hypothesis) -> set[str]:
    found = {m.group(0).lower() for m in _PERSON_RE.finditer(h.statement)}
    for e in h.entities:
        if _PERSON_RE.search(e) or (e[:1].isupper() and " " in e):
            found.add(e.lower())
    for slot, val in h.fillers:
        if slot in {"culprit", "entity"} and val.strip():
            found.add(val.lower())
    return found


def covers(hypothesis: Hypothesis, gap: Gap) -> bool:
    """A slot is covered when an admitted hypothesis fills it."""
    fillers = hypothesis.filler_map()
    if gap.slot in fillers and str(fillers[gap.slot]).strip():
        return True
    if hypothesis.slot == gap.slot and hypothesis.statement.strip():
        return True

    if gap.slot == "culprit":
        return bool(_hyp_people(hypothesis) or fillers.get("culprit"))
    if gap.slot == "location":
        return bool(_hyp_places(hypothesis) or fillers.get("location"))
    if gap.slot == "time":
        return bool(_hyp_times(hypothesis) or fillers.get("time"))
    if gap.slot == "object":
        obj = fillers.get("object", "")
        if obj:
            return True
        low = hypothesis.statement.lower()
        return any(w in low for w in ("ledger", "coin", "item", "object", "book", "gem"))
    if gap.slot == "motive":
        return "motive" in fillers or any(
            w in hypothesis.statement.lower()
            for w in ("because", "motive", "in order to", "to sell", "revenge")
        )
    if gap.slot == "entity":
        return bool(hypothesis.entities) or bool(hypothesis.statement.strip())
    return False


def unresolved_gaps(
    gaps: Sequence[Gap],
    admitted: Iterable[Hypothesis],
) -> list[Gap]:
    admitted_list = list(admitted)
    out: list[Gap] = []
    for gap in gaps:
        if not any(covers(h, gap) for h in admitted_list):
            out.append(gap)
    return out


def covered_slots(gaps: Sequence[Gap], admitted: Iterable[Hypothesis]) -> tuple[str, ...]:
    admitted_list = list(admitted)
    return tuple(g.slot for g in gaps if any(covers(h, g) for h in admitted_list))


def is_sufficient(gaps: Sequence[Gap], admitted: Iterable[Hypothesis]) -> bool:
    required = [g for g in gaps if g.required]
    if not required:
        return bool(list(admitted))
    return not unresolved_gaps(required, admitted)
