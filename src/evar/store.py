"""Immutable-once evidence store of source-linked atomic claims."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from evar.types import AtomicClaim, FrozenStoreError, Polarity


class EvidenceStore:
    """Provenance-preserving store. ``lock()`` freezes it for the rest of the run.

    After lock, ``add`` / ``remove`` raise ``FrozenStoreError``. Downstream
    reasoning may *read* the store; it may never rewrite the evidential basis.
    That is the paper's point: refinement cannot redefine what counts as evidence.
    """

    def __init__(self, claims: Iterable[AtomicClaim] | None = None) -> None:
        self._claims: list[AtomicClaim] = list(claims or [])
        self._locked: bool = False
        self._index: dict[str, AtomicClaim] = {c.id: c for c in self._claims}

    @property
    def locked(self) -> bool:
        return self._locked

    def __len__(self) -> int:
        return len(self._claims)

    def __iter__(self) -> Iterator[AtomicClaim]:
        return iter(self._claims)

    def claims(self) -> tuple[AtomicClaim, ...]:
        return tuple(self._claims)

    def add(self, claim: AtomicClaim) -> None:
        if self._locked:
            raise FrozenStoreError("cannot add to a locked evidence store")
        if claim.id in self._index:
            raise ValueError(f"duplicate claim id: {claim.id}")
        self._claims.append(claim)
        self._index[claim.id] = claim

    def extend(self, claims: Iterable[AtomicClaim]) -> None:
        for claim in claims:
            self.add(claim)

    def remove(self, claim_id: str) -> None:
        if self._locked:
            raise FrozenStoreError("cannot remove from a locked evidence store")
        if claim_id not in self._index:
            raise KeyError(claim_id)
        self._claims = [c for c in self._claims if c.id != claim_id]
        del self._index[claim_id]

    def lock(self) -> None:
        self._locked = True

    def get(self, claim_id: str) -> AtomicClaim | None:
        return self._index.get(claim_id)

    def lookup(
        self,
        *,
        entity: str | None = None,
        predicate: str | None = None,
        polarity: Polarity | None = None,
        ok_only: bool = False,
        exclude_rumor: bool = False,
    ) -> tuple[AtomicClaim, ...]:
        """Return claims matching lightweight metadata filters."""
        entity_l = entity.lower() if entity else None
        pred_l = predicate.lower() if predicate else None
        out: list[AtomicClaim] = []
        for claim in self._claims:
            if ok_only and claim.status.value != "OK":
                continue
            if exclude_rumor and claim.is_rumor:
                continue
            if polarity is not None and claim.polarity != polarity:
                continue
            if entity_l is not None:
                hay = " ".join(claim.entities).lower() + " " + claim.text.lower()
                if entity_l not in hay:
                    continue
            if pred_l is not None:
                hay_p = " ".join(claim.predicates).lower() + " " + claim.text.lower()
                if pred_l not in hay_p:
                    continue
            out.append(claim)
        return tuple(out)

    def ok_claims(self) -> Sequence[AtomicClaim]:
        return tuple(
            c
            for c in self._claims
            if c.status.value == "OK" and not c.is_rumor
        )
