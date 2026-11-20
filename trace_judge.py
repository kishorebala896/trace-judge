"""trace-judge: score the trace, not just the answer.

An agent run is a list of steps: thinks, tool calls, tool results, and a
final answer. Most evals only check the answer. This scores how the agent
*got* there, because in production the trace is where the money, the
hallucinations, and the silent failures live.

Five small judges, all deterministic, no model calls:

  outcome       did the final answer match the expected one?
  grounding     every number in the answer — is it in a tool result, in
                the task itself, or shown as a derivation that checks out?
  tool_use      duplicate calls, ignored errors, empty results treated
                as success
  verification  did the agent re-check anything before answering?
  economy       calls / tokens / estimated cost vs the task's par

A trace is just JSON (see examples/). Run `python demo.py` for the
comparison that motivated this: three agents, same tasks, same final
answers for two of them — and very different scores.
"""

import difflib
import json
import re
import sys
from dataclasses import dataclass, field

# placeholder rates, $ per million tokens — swap for whatever you actually pay
IN_RATE = 3.0
OUT_RATE = 15.0

# numbers too small to mean anything as a quoted fact ("3 invoices")
_IGNORE_BELOW = 10

_NUMBER_RE = re.compile(r"\$?(\d[\d,]*(?:\.\d+)?)")
_DERIVATION_RE = re.compile(
    r"([\d][\d,]*(?:\.\d+)?(?:\s*\+\s*[\d][\d,]*(?:\.\d+)?)+)\s*=\s*\$?([\d][\d,]*(?:\.\d+)?)"
)
_VERIFY_WORDS = ("check", "verify", "confirm", "double-check", "re-check", "recheck")


@dataclass
class Finding:
    judge: str
    text: str


@dataclass
class JudgeResult:
    name: str
    score: float  # 0..1
    detail: str
    findings: list = field(default_factory=list)


def _num(tok):
    return float(tok.replace(",", "").lstrip("$"))


def _numbers(text):
    """All numbers in a string, as floats, skipping trivial small ints."""
    out = []
    for m in _NUMBER_RE.finditer(text or ""):
        try:
            v = _num(m.group(1))
        except ValueError:
            continue
        if v < _IGNORE_BELOW and "." not in m.group(1) and "$" not in m.group(0):
            continue
        out.append(v)
    return out


def _norm_answer(text):
    return re.sub(r"[^0-9a-z.]", "", (text or "").lower())


class Trace:
    def __init__(self, raw):
        self.raw = raw
        self.task = raw.get("task", "")
        self.task_id = raw.get("task_id", "untitled")
        self.expected = raw.get("expected_answer")
        self.par_calls = raw.get("par_calls")
        self.steps = raw.get("steps", [])

    def final_answer(self):
        for s in reversed(self.steps):
            if s.get("kind") == "answer":
                return s.get("text", "")
        return ""

    def tool_results_text(self):
        return "\n".join(
            str(s.get("result", "")) for s in self.steps if s.get("kind") == "tool_result"
        )

    def tool_calls(self):
        return [s for s in self.steps if s.get("kind") == "tool_call"]

    def tokens(self):
        tin = sum(s.get("tokens_in", 0) for s in self.steps)
        tout = sum(s.get("tokens_out", 0) for s in self.steps)
        return tin, tout

    def cost(self):
        tin, tout = self.tokens()
        return tin / 1e6 * IN_RATE + tout / 1e6 * OUT_RATE


def judge_outcome(trace):
    if trace.expected is None:
        return JudgeResult("outcome", 1.0, "no expected answer given, skipped")
    got = _norm_answer(trace.final_answer())
    want = _norm_answer(str(trace.expected))
    if not got:
        return JudgeResult("outcome", 0.0, "no final answer in trace",
                           [Finding("outcome", "trace ends without an answer")])
    if want and want in got:
        return JudgeResult("outcome", 1.0, f"answer contains expected '{trace.expected}'")
    return JudgeResult(
        "outcome", 0.0, f"answer does not contain expected '{trace.expected}'",
        [Finding("outcome", f"expected '{trace.expected}', got '{trace.final_answer()}'")],
    )


def judge_grounding(trace):
    answer = trace.final_answer()
    if not answer:
        return JudgeResult("grounding", 0.0, "no answer to ground",
                           [Finding("grounding", "no final answer")])
    known = set(_numbers(trace.task)) | set(_numbers(trace.tool_results_text()))
    # derivations the answer shows its working for: "a + b = c"
    derived_ok = set()
    for m in _DERIVATION_RE.finditer(answer):
        parts = [_num(p) for p in re.split(r"\s*\+\s*", m.group(1))]
        total = _num(m.group(2))
        if all(p in known for p in parts) and abs(sum(parts) - total) < 0.01:
            derived_ok.add(total)
    claims = _numbers(answer)
    if not claims:
        return JudgeResult("grounding", 1.0, "no numeric claims in answer")
    bad = [c for c in claims if c not in known and c not in derived_ok]
    good = len(claims) - len(bad)
    findings = [
        Finding("grounding",
                f"{c:g} appears in the answer but in no tool result, "
                "the task, or a checked derivation — invented or unverified")
        for c in bad
    ]
    return JudgeResult(
        "grounding", good / len(claims),
        f"{good}/{len(claims)} numeric claims trace back to a tool result, "
        "the task, or a verified derivation",
        findings,
    )


def judge_tool_use(trace):
    findings = []
    seen = {}
    for i, s in enumerate(trace.steps):
        if s.get("kind") != "tool_call":
            continue
        key = (s.get("tool"), json.dumps(s.get("args", {}), sort_keys=True))
        if key in seen:
            # a repeat is fine if the first attempt failed — that's a retry
            first = seen[key]
            failed_first = any(
                r.get("kind") == "tool_result" and r.get("tool") == s.get("tool")
                and (not r.get("ok", True) or not str(r.get("result", "")).strip())
                for r in trace.steps[first:i]
            )
            if not failed_first:
                findings.append(Finding(
                    "tool_use",
                    f"step {i}: repeat call to {s.get('tool')} with identical "
                    f"args (first at step {first}) — result already in hand"))
        else:
            seen[key] = i
    # ignored errors: a failed/empty result with no retry of the same
    # call and no acknowledgement, then an answer anyway
    for i, s in enumerate(trace.steps):
        if s.get("kind") != "tool_result":
            continue
        ok = s.get("ok", True)
        empty = not str(s.get("result", "")).strip()
        if not ok or empty:
            # the call this result belongs to: nearest earlier call, same tool
            call_args = None
            for prev in reversed(trace.steps[:i]):
                if prev.get("kind") == "tool_call" and prev.get("tool") == s.get("tool"):
                    call_args = prev.get("args", {})
                    break
            rest = trace.steps[i + 1:]
            # a retry is the same call again — allowing for a typo fix in
            # the args, e.g. a corrected search query. Same tool, different
            # id ("INV-103" after "INV-102" failed) is NOT a retry: the
            # failed lookup's answer never came back.
            failed_args = json.dumps(call_args, sort_keys=True)
            retried = any(
                r.get("kind") == "tool_call" and r.get("tool") == s.get("tool")
                and difflib.SequenceMatcher(
                    None, failed_args,
                    json.dumps(r.get("args", {}), sort_keys=True)).ratio() >= 0.9
                for r in rest
            )
            acknowledged = any(
                r.get("kind") == "think"
                and ("fail" in (r.get("text") or "").lower()
                     or "error" in (r.get("text") or "").lower())
                for r in rest
            )
            answered_anyway = any(r.get("kind") == "answer" for r in rest)
            if not retried and not acknowledged and answered_anyway:
                why = "failed" if not ok else "came back empty"
                findings.append(Finding(
                    "tool_use",
                    f"step {i}: {s.get('tool')} {why} and the agent "
                    "answered anyway — silent failure, no retry"))
    score = max(0.0, 1.0 - 0.25 * len(findings))
    return JudgeResult(
        "tool_use", score,
        f"{len(trace.tool_calls())} tool calls, {len(findings)} issue(s)",
        findings,
    )


def judge_verification(trace):
    answer_idx = next(
        (i for i, s in enumerate(trace.steps) if s.get("kind") == "answer"), None
    )
    if answer_idx is None:
        return JudgeResult("verification", 0.0, "no answer",
                           [Finding("verification", "trace ends without an answer")])
    before = trace.steps[:answer_idx]
    checked = any(
        s.get("kind") == "think"
        and any(w in (s.get("text") or "").lower() for w in _VERIFY_WORDS)
        for s in before
    )
    shown_work = bool(_DERIVATION_RE.search(trace.final_answer()))
    if checked:
        return JudgeResult("verification", 1.0, "explicit re-check before answering")
    if shown_work:
        return JudgeResult(
            "verification", 0.6, "shows its working but never re-checks it",
            [Finding("verification", "derivation shown, no explicit re-check step")])
    return JudgeResult(
        "verification", 0.2, "answered straight from the first results",
        [Finding("verification", "no re-check step before the answer")],
    )


def judge_economy(trace):
    calls = len(trace.tool_calls())
    tin, tout = trace.tokens()
    if trace.par_calls:
        score = min(1.0, trace.par_calls / max(calls, 1))
        detail = f"{calls} calls vs par {trace.par_calls}, " \
                 f"{tin + tout} tokens, ${trace.cost():.4f}"
    else:
        score, detail = 1.0, f"{calls} calls, {tin + tout} tokens, " \
                              f"${trace.cost():.4f} (no par given)"
    return JudgeResult("economy", score, detail)


WEIGHTS = {"outcome": 0.30, "grounding": 0.25, "tool_use": 0.20,
           "verification": 0.15, "economy": 0.10}

JUDGES = [judge_outcome, judge_grounding, judge_tool_use,
          judge_verification, judge_economy]


def evaluate(trace):
    results = [j(trace) for j in JUDGES]
    overall = sum(WEIGHTS[r.name] * r.score for r in results)
    by_name = {r.name: r for r in results}
    if by_name["outcome"].score == 0 or by_name["grounding"].score < 0.5:
        verdict = "FAIL"
    elif overall < 0.8:
        verdict = "PASS WITH ISSUES"
    else:
        verdict = "PASS"
    return {"trace": trace, "results": results, "overall": overall, "verdict": verdict}


def render(report, show_findings=True):
    t = report["trace"]
    lines = [f"== {t.task_id}: {report['verdict']}  (overall {report['overall']:.2f}) =="]
    for r in report["results"]:
        lines.append(f"  {r.name:<13} {r.score:.2f}  {r.detail}")
    if show_findings:
        for r in report["results"]:
            for f in r.findings:
                lines.append(f"    ! [{f.judge}] {f.text}")
    tin, tout = t.tokens()
    lines.append(f"  cost: {len(t.tool_calls())} calls, {len(t.steps)} steps, "
                 f"{tin + tout} tokens, ${t.cost():.4f}")
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print("usage: python trace_judge.py <trace.json> [trace2.json ...]")
        print("       python trace_judge.py --demo  (runs demo.py)")
        return 1
    for path in argv[1:]:
        with open(path) as f:
            report = evaluate(Trace(json.load(f)))
        print(render(report))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
