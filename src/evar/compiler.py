"""Compile a narrative into source-linked atomic claims.

The compiler is deliberately lexical — no LLM, no GPU. Each sentence yields
at least one claim (sentences are never dropped). Offsets slice back into
the original narrative. Metadata (entities, times, places, polarity, rumor
and uncertainty tags) is recovered from the cited span only.
"""

from __future__ import annotations

import re

from evar.types import AtomicClaim, ClaimStatus, Polarity, SourceSpan

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

_STOP_TITLE = {
    "the", "a", "an", "on", "in", "at", "to", "of", "for", "and", "or", "but",
    "with", "from", "by", "as", "if", "so", "no", "not", "nor", "yet", "both",
    "each", "every", "some", "any", "all", "this", "that", "these", "those",
    "he", "she", "it", "they", "we", "you", "his", "her", "its", "their",
    "who", "whom", "whose", "when", "where", "why", "how", "what", "which",
    "two", "three", "four", "five", "six", "ten", "after", "before", "during",
    "later", "then", "once", "rain", "whoever", "someone", "anyone", "nobody",
    "morning", "evening", "night", "dusk", "dawn", "noon", "midnight",
}

_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

_PLACE_KEYWORDS = {
    "office", "harbor", "harbour", "ferry", "warehouse", "locker", "pier",
    "dock", "drawer", "counter", "bay", "desk", "camera", "kitchen", "park",
    "street", "alley", "wharf", "customs", "loading", "rack", "doors",
    "cabin", "bridge", "yard", "gate", "room", "hall", "station", "bank",
    "library", "garden", "basement", "attic", "shop", "market", "plaza",
}

_RUMOR_CUES = (
    "rumor", "rumour", "rumored", "rumoured", "allegedly", "supposedly",
    "claimed that", "hearsay", "gossip", "word is", "people say",
)

_UNCERTAIN_CUES = (
    "illegible", "smeared", "unclear", "unknown", "cannot be read",
    "can't be read", "maybe", "perhaps", "uncertain", "unreadable",
    "indecipherable", "blurred",
)

_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|nobody|none|neither|nor|nothing|"
    r"didn't|didnt|wasn't|wasnt|isn't|isnt|aren't|arent|"
    r"haven't|havent|hasn't|hasnt|wouldn't|wouldnt|"
    r"couldn't|couldnt|won't|wont|cannot|can't)\b",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"\b(?:\d{1,2}:\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?|"
    r"\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+night|\s+morning|\s+evening|\s+afternoon)?|"
    r"noon|midnight|dusk|dawn)\b",
    re.IGNORECASE,
)

_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_QUOTE_RE = re.compile(r"[\"\u201c\u201d]([^\"\u201c\u201d]+)[\"\u201c\u201d]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201c(\[])")
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*;\s*|\s+but\s+|\s+however,?\s+|\s+although\s+|\s+yet\s+",
    re.IGNORECASE,
)

# Light verb list used only to decide whether an "and"-split is safe.
_VERBISH_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"clocked|seen|stole|stolen|found|shows|showed|claimed|issues|"
    r"issued|entered|enter|used|use|sold|sell|cut|held|holds|"
    r"returned|return|left|leave|swore|told|wrote|keeps|keep|"
    r"recorded|discovered|gone|forced|recovered|wearing|wore)\b",
    re.IGNORECASE,
)

_IRREGULAR_STEMS: dict[str, str] = {
    "stolen": "steal",
    "stole": "steal",
    "sold": "sell",
    "found": "find",
    "saw": "see",
    "seen": "see",
    "went": "go",
    "made": "make",
    "cut": "cut",
    "kept": "keep",
    "held": "hold",
    "shown": "show",
    "shows": "show",
    "showed": "show",
    "entered": "enter",
    "entering": "enter",
    "clocked": "clock",
    "claimed": "claim",
    "issued": "issue",
    "issues": "issue",
    "smeared": "smear",
    "forced": "force",
    "recovered": "recover",
    "returned": "return",
    "wearing": "wear",
    "wore": "wear",
}


def stem(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[w]
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w


def _is_rumor(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in _RUMOR_CUES)


def _is_uncertain(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in _UNCERTAIN_CUES)


def _polarity(text: str) -> Polarity:
    return Polarity.NEGATED if _NEGATION_RE.search(text) else Polarity.ASSERTED


def extract_times(text: str) -> tuple[str, ...]:
    found = [m.group(0).strip() for m in _TIME_RE.finditer(text)]
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in found:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return tuple(out)


def extract_quotes(text: str) -> tuple[str, ...]:
    return tuple(m.group(1).strip() for m in _QUOTE_RE.finditer(text) if m.group(1).strip())


def extract_names(text: str) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for m in _NAME_RE.finditer(text):
        raw = m.group(1).strip()
        parts = raw.split()
        if any(p.lower() in _DAYS or p.lower() in _MONTHS for p in parts):
            continue
        if all(p.lower() in _STOP_TITLE for p in parts):
            continue
        if len(parts) == 1 and parts[0].lower() in _STOP_TITLE:
            continue
        if len(parts) == 1 and parts[0].lower() in _PLACE_KEYWORDS:
            continue
        key = raw.lower()
        if key not in seen:
            seen.add(key)
            names.append(raw)
    return tuple(names)


def extract_places(text: str) -> tuple[str, ...]:
    low = text.lower()
    places: list[str] = []
    seen: set[str] = set()

    # multi-word place phrases first
    phrase_res = [
        r"\bcustoms office\b",
        r"\bsouth ferry\b",
        r"\bnorth ferry\b",
        r"\bloading bay\b",
        r"\bbay doors\b",
        r"\boffice counter\b",
        r"\bclerk'?s desk\b",
        r"pier\s+\d+",
        r"\bthe warehouse\b",
        r"\bthe office\b",
        r"\bthe kitchen\b",
        r"\bthe park\b",
        r"\bthe harbor\b",
        r"\bthe locker\b",
    ]
    for pat in phrase_res:
        for m in re.finditer(pat, low):
            span = text[m.start() : m.end()].strip()
            key = span.lower()
            if key not in seen:
                seen.add(key)
                places.append(span)

    for kw in sorted(_PLACE_KEYWORDS, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(kw)}\b", low):
            span = text[m.start() : m.end()]
            key = span.lower()
            if key not in seen:
                seen.add(key)
                places.append(span)
    return tuple(places)


_PREDICATE_STOP = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "from", "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "that", "this", "these", "those",
    "who", "whom", "which", "what", "when", "where", "why", "how", "his", "her",
    "its", "their", "our", "your", "not", "never", "no", "only", "also", "then",
    "than", "into", "over", "after", "before", "during", "until", "while",
}


def extract_predicates(text: str, entities: tuple[str, ...]) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    entity_tokens: set[str] = set()
    for ent in entities:
        entity_tokens.update(re.findall(r"[a-z']+", ent.lower()))
    preds: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _PREDICATE_STOP or tok in entity_tokens:
            continue
        if len(tok) < 3:
            continue
        s = stem(tok)
        if s not in seen:
            seen.add(s)
            preds.append(s)
    return tuple(preds)


def _split_sentences(narrative: str) -> list[str]:
    """Split on sentence boundaries, then on leftover title/line breaks."""
    if not (narrative or "").strip():
        return []
    blocks = re.split(r"\n\s*\n", narrative.strip())
    sentences: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        chunks = [c.strip() for c in _SENTENCE_RE.split(block) if c.strip()]
        if not chunks:
            chunks = [block]
        for chunk in chunks:
            for line in chunk.split("\n"):
                line = line.strip()
                if line:
                    sentences.append(line)
    return sentences


def _split_clauses(sentence: str) -> list[str]:
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence) if p.strip()]
    if len(parts) <= 1:
        # try "and" split only when both sides look clausal
        and_parts = [p.strip() for p in re.split(r"\s+and\s+", sentence) if p.strip()]
        if len(and_parts) >= 2 and all(_VERBISH_RE.search(p) for p in and_parts):
            return and_parts
        return [sentence.strip()]
    return parts


def _locate(narrative: str, snippet: str, cursor: int) -> SourceSpan:
    idx = narrative.find(snippet, cursor)
    if idx < 0:
        idx = narrative.find(snippet.strip(), cursor)
    if idx < 0:
        idx = narrative.find(snippet)
    if idx < 0:
        # last resort: keep a zero-width-safe span at cursor
        idx = min(cursor, len(narrative))
        end = min(idx + len(snippet), len(narrative))
        return SourceSpan(start=idx, end=end, text=narrative[idx:end] or snippet)
    end = idx + len(snippet)
    return SourceSpan(start=idx, end=end, text=narrative[idx:end])


def _entities_for(text: str) -> tuple[str, ...]:
    names = extract_names(text)
    quotes = extract_quotes(text)
    places = extract_places(text)
    times = extract_times(text)
    merged: list[str] = []
    seen: set[str] = set()
    for item in (*names, *quotes, *places, *times):
        key = item.lower()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return tuple(merged)


def _tag_claim(text: str) -> tuple[ClaimStatus, int, str, bool]:
    rumor = _is_rumor(text)
    uncertain = _is_uncertain(text)
    if rumor:
        return ClaimStatus.UNCERTAIN, 2, "rumor-attributed; not treated as fact", True
    if uncertain:
        return ClaimStatus.UNCERTAIN, 1, "local uncertainty cue", False
    return ClaimStatus.OK, 0, "", False


def _retag_conflicts(claims: list[AtomicClaim]) -> list[AtomicClaim]:
    """Mark claims that share a person-entity but disagree on place or polarity."""
    # Person-like: two-token capitalized names
    def people(c: AtomicClaim) -> set[str]:
        out: set[str] = set()
        for e in c.entities:
            if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", e):
                out.add(e.lower())
        return out

    conflict_ids: set[str] = set()
    notes: dict[str, str] = {}
    for i, a in enumerate(claims):
        pa = people(a)
        if not pa:
            continue
        for b in claims[i + 1 :]:
            pb = people(b)
            shared = pa & pb
            if not shared:
                continue
            pred_overlap = set(a.predicates) & set(b.predicates)
            if a.polarity != b.polarity and pred_overlap:
                conflict_ids.add(a.id)
                conflict_ids.add(b.id)
                notes[a.id] = "opposite polarity on overlapping person+predicate"
                notes[b.id] = "opposite polarity on overlapping person+predicate"
                continue
            places_a = {p.lower() for p in a.places}
            places_b = {p.lower() for p in b.places}
            if (
                a.polarity == Polarity.ASSERTED
                and b.polarity == Polarity.ASSERTED
                and places_a
                and places_b
                and places_a.isdisjoint(places_b)
            ):
                # exclusive asserted locations for the same person — flag, do not drop
                conflict_ids.add(a.id)
                conflict_ids.add(b.id)
                notes[a.id] = "exclusive locations for the same person"
                notes[b.id] = "exclusive locations for the same person"

    retagged: list[AtomicClaim] = []
    for c in claims:
        if c.id in conflict_ids and c.status == ClaimStatus.OK:
            retagged.append(
                AtomicClaim(
                    id=c.id,
                    text=c.text,
                    source=c.source,
                    entities=c.entities,
                    polarity=c.polarity,
                    predicates=c.predicates,
                    status=ClaimStatus.CONFLICT,
                    severity=max(c.severity, 2),
                    note=notes.get(c.id, c.note),
                    times=c.times,
                    places=c.places,
                    is_rumor=c.is_rumor,
                )
            )
        else:
            retagged.append(c)
    return retagged


def compile_narrative(narrative: str) -> list[AtomicClaim]:
    """Sentence-split ``narrative`` into source-linked atomic claims.

    Never silently drops a sentence: if clause splitting fails, the whole
    sentence is kept as a single claim.
    """
    if not (narrative or "").strip():
        return []

    sentences = _split_sentences(narrative)
    claims: list[AtomicClaim] = []
    cursor = 0
    seq = 0

    for sent in sentences:
        span_sent = _locate(narrative, sent, cursor)
        cursor = span_sent.end
        clauses = _split_clauses(sent)
        # If splitting produced nothing, keep the sentence.
        if not clauses:
            clauses = [sent]
        inner_cursor = span_sent.start
        for clause in clauses:
            clause = clause.strip(" \t,;")
            if not clause:
                # still emit a claim so the parent sentence is not dropped
                clause = sent.strip()
            span = _locate(narrative, clause, inner_cursor)
            inner_cursor = max(inner_cursor, span.end)
            seq += 1
            entities = _entities_for(clause)
            times = extract_times(clause)
            places = extract_places(clause)
            predicates = extract_predicates(clause, entities)
            status, severity, note, rumor = _tag_claim(clause)
            claims.append(
                AtomicClaim(
                    id=f"c{seq:03d}",
                    text=clause,
                    source=span,
                    entities=entities,
                    polarity=_polarity(clause),
                    predicates=predicates,
                    status=status,
                    severity=severity,
                    note=note,
                    times=times,
                    places=places,
                    is_rumor=rumor,
                )
            )

    return _retag_conflicts(claims)
