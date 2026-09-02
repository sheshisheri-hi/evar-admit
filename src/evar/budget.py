"""Instance-specific inference budget from unresolved gaps and uncertainty."""

from __future__ import annotations

from dataclasses import dataclass


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@dataclass
class Budget:
    """Refinement budget.

    Spec used here (reference implementation, not the paper's Gamma/tau formula):
        max_rounds = clamp(1 + n_unresolved_gaps, 1, 6)
        max_hypotheses_per_round = 3
    Each proposed hypothesis costs 1. Uncertainty is recorded and shown, and
    can bump remaining-hypothesis headroom by one extra candidate when the
    store is noisy, without exceeding the hard round cap.
    """

    max_rounds: int
    max_hypotheses_per_round: int
    max_hypotheses: int
    n_gaps: int
    n_uncertain: int
    rounds_used: int = 0
    hypotheses_used: int = 0

    @classmethod
    def from_signals(
        cls,
        n_unresolved_gaps: int,
        n_uncertain: int = 0,
        *,
        max_rounds: int | None = None,
        max_hypotheses_per_round: int = 3,
    ) -> "Budget":
        n_gaps = max(0, int(n_unresolved_gaps))
        n_unc = max(0, int(n_uncertain))
        if max_rounds is None:
            computed = clamp(1 + n_gaps, 1, 6)
        else:
            computed = max(0, int(max_rounds))
        per_round = max(1, int(max_hypotheses_per_round))
        extra = 1 if n_unc > 0 else 0
        max_hyps = computed * per_round + extra
        return cls(
            max_rounds=computed,
            max_hypotheses_per_round=per_round,
            max_hypotheses=max_hyps,
            n_gaps=n_gaps,
            n_uncertain=n_unc,
        )

    def remaining_rounds(self) -> int:
        return max(0, self.max_rounds - self.rounds_used)

    def remaining_hypotheses(self) -> int:
        return max(0, self.max_hypotheses - self.hypotheses_used)

    def can_continue(self) -> bool:
        return self.remaining_rounds() > 0 and self.remaining_hypotheses() > 0

    def consume_round(self) -> None:
        self.rounds_used += 1

    def consume_hypotheses(self, n: int) -> None:
        self.hypotheses_used += max(0, n)

    def snapshot(self) -> dict[str, int]:
        return {
            "max_rounds": self.max_rounds,
            "max_hypotheses_per_round": self.max_hypotheses_per_round,
            "max_hypotheses": self.max_hypotheses,
            "n_gaps": self.n_gaps,
            "n_uncertain": self.n_uncertain,
            "rounds_used": self.rounds_used,
            "hypotheses_used": self.hypotheses_used,
            "remaining_rounds": self.remaining_rounds(),
            "remaining_hypotheses": self.remaining_hypotheses(),
        }
