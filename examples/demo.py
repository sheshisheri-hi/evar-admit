#!/usr/bin/env python3
"""The Harbor Watch — an offline EVAR session over an original mock mystery."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/demo.py` without installing, as well as PYTHONPATH=src.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evar.engine import EVAREngine
from evar.types import Verdict

NARRATIVE = """
The Harbor Watch

On Wednesday morning the customs office on Pier 4 discovered that the harbor ledger was gone. The bound book, which recorded every inbound crate from Tuesday's tide, had been stolen Tuesday night from the locked drawer of the clerk's desk in the customs office.

Mara Cole clocked out at 9:10pm. Two passengers later told the harbor master they saw Mara Cole on the south ferry at 9:40pm, her blue slicker bright under the deck lamps. Mara Cole did not return to the pier that night. Mara Cole's slicker is blue.

Jonas Pike, the dock foreman, held the only spare office key. Jonas Pike was in the warehouse until 11pm. Two coworkers swore Jonas Pike never left the loading bay. Jonas Pike was not seen near the customs office after dusk.

A rumor circulating among the night crew claimed that Mara Cole had sold ledgers before. No one offered a date, a buyer, or a document. The harbor master wrote the rumor down as rumor, not as fact.

Rain had blown in from the channel and smeared the sign-in book on the office counter. The 10:00pm line is illegible. Whoever signed, or did not sign, cannot be read.

On Wednesday a spare key was found in Jonas Pike's locker. The locker's padlock was cut. The key itself was the office spare, but the cut lock meant anyone with a bolt cutter could have planted it there after the theft.

A customs drone frame, time-stamped 10:22pm, shows a figure in a yellow slicker entering the customs office. The warehouse issues yellow slickers to the night shift and keeps a rack of them by the bay doors. The figure's face is turned from the camera.

The drawer lock was not forced. Whoever entered used a key, or found the drawer already unlocked. The ledger has not been recovered.
""".strip()

QUESTION = "Who stole the harbor ledger, and what evidence actually supports that?"

# ANSI — no extra dependency. Falls back to plain text if stdout isn't a tty.
def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD = lambda s: _c("1", s)
DIM = lambda s: _c("2", s)
CYAN = lambda s: _c("36", s)
GREEN = lambda s: _c("32", s)
YELLOW = lambda s: _c("33", s)
RED = lambda s: _c("31", s)
MAGENTA = lambda s: _c("35", s)


def _verdict_paint(v: Verdict) -> str:
    if v == Verdict.ADMITTED:
        return GREEN(v.value)
    if v == Verdict.DISCARDED:
        return RED(v.value)
    return YELLOW(v.value)


def _hr(char: str = "─", n: int = 72) -> str:
    return DIM(char * n)


def main() -> int:
    words = len(NARRATIVE.split())
    engine = EVAREngine()
    result = engine.run(NARRATIVE, QUESTION)

    print(_hr("═"))
    print(BOLD("  EVAR  ·  Evidence-Validated Hypothesis Admission"))
    print(DIM("  The Harbor Watch  ·  deterministic reasoner  ·  locked store"))
    print(_hr("═"))
    print()
    print(CYAN(BOLD("[1] Compiled evidence store")))
    print(f"    claims : {len(result.claims)}   (immutable after lock)")
    print(f"    words  : {words}   (original narrative)")
    print(f"    uncertain / rumor / conflict units : {result.n_uncertain}")
    print()
    print(DIM("    sample claims with source excerpts:"))
    shown = 0
    for claim in result.claims:
        if shown >= 6:
            break
        excerpt = claim.source.text.replace("\n", " ")
        if len(excerpt) > 88:
            excerpt = excerpt[:85] + "..."
        print(
            f"    {DIM(claim.id)}  {claim.polarity.value:9s}  "
            f"{claim.status.value:10s}  rumor={str(claim.is_rumor).lower()}"
        )
        print(f"         {excerpt}")
        shown += 1
    print()

    print(CYAN(BOLD("[2] Instance-specific budget")))
    print(f"    uncertain / conflict / rumor units : {result.n_uncertain}")
    print(
        f"    max_rounds = clamp(1 + n_unresolved_gaps, 1, 6) = {result.budget_max_rounds}"
    )
    print(f"    max_hypotheses (incl. uncertainty headroom) = {result.budget_max_hypotheses}")
    print("    max_hypotheses_per_round = 3")
    print("    each proposed hypothesis costs 1; stop on sufficiency or remaining=0")
    print()

    print(CYAN(BOLD("[3] Refinement rounds")))
    if not result.rounds:
        print("    (no rounds — empty or fast path)")
    for rnd in result.rounds:
        print(_hr())
        print(
            BOLD(f"    round {rnd.round_index}")
            + DIM(
                f"   remaining rounds={rnd.remaining_rounds}  "
                f"remaining hyps={rnd.remaining_hypotheses}"
            )
        )
        print(DIM(f"    unresolved after round: {', '.join(rnd.unresolved_slots) or '(none)'}"))
        rec_by_id = {v.hypothesis_id: v for v in rnd.verdicts}
        chal_by_id = {c.hypothesis_id: c for c in rnd.challenges}
        for hyp in rnd.proposed:
            rec = rec_by_id[hyp.id]
            chal = chal_by_id[hyp.id]
            print()
            print(f"    {MAGENTA('hyp')} {hyp.id}")
            print(f"         {hyp.statement}")
            print(
                DIM("         challenge.support   ")
                + ", ".join(chal.support_queries[:6])
            )
            print(
                DIM("         challenge.contra    ")
                + ", ".join(chal.contradiction_queries[:3])
            )
            print(
                f"         verdict {_verdict_paint(rec.verdict)}   "
                f"support={rec.support_score:.2f}   "
                f"contradicted={rec.contradicted}"
            )
            if rec.supporting_claim_ids:
                print(DIM(f"         support ids : {', '.join(rec.supporting_claim_ids)}"))
            if rec.contradicting_claim_ids:
                print(DIM(f"         contra  ids : {', '.join(rec.contradicting_claim_ids)}"))
        print(DIM(f"    {rnd.notes}"))
    print()

    print(CYAN(BOLD("[4] Final answer-supporting state")))
    print(_hr())

    def _block(title: str, items: tuple, paint) -> None:
        print(f"    {paint(title)} ({len(items)})")
        if not items:
            print(DIM("         — none —"))
            return
        for h in items:
            print(f"         • {h.statement}")

    _block("ADMITTED", result.admitted, GREEN)
    print()
    _block("QUARANTINED", result.quarantined, YELLOW)
    print()
    _block("DISCARDED", result.discarded, RED)
    print()
    print(
        DIM(
            f"    stop={result.stop_reason}  "
            f"covered={list(result.covered_slots)}  "
            f"unresolved={list(result.unresolved_slots)}"
        )
    )
    print()

    print(CYAN(BOLD("[5] Grounded answer (admitted hypotheses only)")))
    print(_hr())
    print(f"    Q: {QUESTION}")
    print()
    ans = result.answer
    width = 70
    words_out = ans.split()
    line: list[str] = []
    n = 0
    for w in words_out:
        if n + len(w) + 1 > width and line:
            print("    " + " ".join(line))
            line, n = [w], len(w)
        else:
            line.append(w)
            n += len(w) + 1
    if line:
        print("    " + " ".join(line))
    print()
    print(_hr("═"))
    print(DIM("  EVAR reference  ·  Liu, Ji, Ping  ·  arXiv:2608.29835  ·  EMNLP 2026"))
    print(_hr("═"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
