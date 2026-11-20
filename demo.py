"""Demo: three agents, same three tasks, judged on their traces.

The tasks are small on purpose — the point is the comparison, not the
difficulty. Two of the three agents get every answer right. Only the
trace scores tell them apart.

Run it:  python demo.py
"""

import json
from trace_judge import Trace, evaluate, render

LEDGER = {"INV-101": 142.50, "INV-102": 201.30, "INV-103": 142.40}
QUOTES = {"Acme": 89.00, "Beta": 74.50, "Gamma": 92.25}
POLICY = "Refunds are accepted within 30 days of delivery, no restocking fee."

TASKS = {
    "invoice-total": {
        "task": "Add up the totals of invoices INV-101, INV-102 and INV-103.",
        "expected_answer": "486.20",
        "par_calls": 3,
    },
    "cheapest-quote": {
        "task": "Find the cheapest of the three vendor quotes (Acme, Beta, Gamma).",
        "expected_answer": "74.50",
        "par_calls": 3,
    },
    "refund-policy": {
        "task": "A customer bought something 20 days ago. Can they return it, "
                "and is there a fee?",
        "expected_answer": "Yes",
        "par_calls": 1,
    },
}

# each step is a dict; "t" is seconds in, token counts are made up but
# shaped like a real run (a lookup costs a few hundred tokens in+out)
def call(t, tool, args, tin=180, tout=90):
    return {"t": t, "kind": "tool_call", "tool": tool, "args": args,
            "tokens_in": tin, "tokens_out": tout}


def result(t, tool, text, ok=True, tin=60, tout=120):
    return {"t": t, "kind": "tool_result", "tool": tool, "ok": ok,
            "result": text, "tokens_in": tin, "tokens_out": tout}


def think(t, text, tin=120, tout=60):
    return {"t": t, "kind": "think", "text": text,
            "tokens_in": tin, "tokens_out": tout}


def answer(t, text, tin=200, tout=80):
    return {"t": t, "kind": "answer", "text": text,
            "tokens_in": tin, "tokens_out": tout}


def careful(task_id):
    """Does the boring right thing: looks everything up, shows the math,
    re-checks before answering."""
    if task_id == "invoice-total":
        steps = [
            think(0.0, "Need all three invoice totals, then add them."),
            call(0.4, "ledger.lookup", {"invoice": "INV-101"}),
            result(1.0, "ledger.lookup", "INV-101 total $142.50"),
            call(1.4, "ledger.lookup", {"invoice": "INV-102"}),
            result(2.0, "ledger.lookup", "INV-102 total $201.30"),
            call(2.4, "ledger.lookup", {"invoice": "INV-103"}),
            result(3.0, "ledger.lookup", "INV-103 total $142.40"),
            think(3.4, "Let me verify the addition before I answer: "
                       "142.50 + 201.30 + 142.40 = 486.20. Checks out."),
            answer(3.8, "The three invoices total $486.20 "
                         "(142.50 + 201.30 + 142.40 = 486.20)."),
        ]
    elif task_id == "cheapest-quote":
        steps = [
            think(0.0, "Need all three quotes before I can pick the cheapest."),
            call(0.4, "quotes.get", {"vendor": "Acme"}),
            result(1.0, "quotes.get", "Acme quote $89.00"),
            call(1.4, "quotes.get", {"vendor": "Beta"}),
            result(2.0, "quotes.get", "Beta quote $74.50"),
            call(2.4, "quotes.get", {"vendor": "Gamma"}),
            result(3.0, "quotes.get", "Gamma quote $92.25"),
            think(3.4, "Let me double-check: 74.50 < 89.00 and 74.50 < 92.25, "
                       "so Beta is cheapest."),
            answer(3.8, "Cheapest is Beta at $74.50."),
        ]
    else:
        steps = [
            think(0.0, "Need the actual refund policy, not my guess."),
            call(0.4, "docs.search", {"query": "refund policy"}),
            result(1.0, "docs.search", POLICY),
            think(1.4, "Let me confirm: 20 days is within 30 days, and the "
                       "policy says no restocking fee."),
            answer(1.8, "Yes — returns are accepted within 30 days of delivery "
                        "and there is no restocking fee."),
        ]
    return dict(TASKS[task_id], task_id=task_id, steps=steps)


def sloppy(task_id):
    """Gets the answers right and the trace wrong: repeats a call, invents
    a number, never re-checks."""
    if task_id == "invoice-total":
        steps = [
            think(0.0, "Look up the invoices."),
            call(0.4, "ledger.lookup", {"invoice": "INV-101"}),
            result(1.0, "ledger.lookup", "INV-101 total $142.50"),
            call(1.4, "ledger.lookup", {"invoice": "INV-101"}),  # already have this
            result(2.0, "ledger.lookup", "INV-101 total $142.50"),
            call(2.4, "ledger.lookup", {"invoice": "INV-102"}),
            result(3.0, "ledger.lookup", "INV-102 total $201.30"),
            call(3.4, "ledger.lookup", {"invoice": "INV-103"}),
            result(4.0, "ledger.lookup", "INV-103 total $142.40"),
            answer(4.4, "The three invoices total $486.20."),
        ]
    elif task_id == "cheapest-quote":
        steps = [
            call(0.4, "quotes.get", {"vendor": "Acme"}),
            result(1.0, "quotes.get", "Acme quote $89.00"),
            call(1.4, "quotes.get", {"vendor": "Beta"}),
            result(2.0, "quotes.get", "Beta quote $74.50"),
            call(2.4, "quotes.get", {"vendor": "Gamma"}),
            result(3.0, "quotes.get", "Gamma quote $92.25"),
            answer(3.4, "Cheapest is Beta at $74.50, plus $4.99 shipping."),
        ]
    else:
        steps = [
            call(0.4, "docs.search", {"query": "refund polciy"}),  # typo'd query
            result(1.0, "docs.search", "", ok=False),  # error, ignored
            call(1.4, "docs.search", {"query": "refund policy"}),
            result(2.0, "docs.search", POLICY),
            answer(2.4, "Yes — returns are accepted within 30 days of delivery "
                        "and there is no restocking fee."),
        ]
    return dict(TASKS[task_id], task_id=task_id, steps=steps)


def unlucky(task_id):
    """A failed lookup it never notices, and a confident wrong answer."""
    if task_id == "invoice-total":
        steps = [
            call(0.4, "ledger.lookup", {"invoice": "INV-101"}),
            result(1.0, "ledger.lookup", "INV-101 total $142.50"),
            call(1.4, "ledger.lookup", {"invoice": "INV-102"}),
            result(2.0, "ledger.lookup", "ledger timeout", ok=False),
            call(2.4, "ledger.lookup", {"invoice": "INV-103"}),
            result(3.0, "ledger.lookup", "INV-103 total $142.40"),
            answer(3.4, "The three invoices total $420.00."),
        ]
    elif task_id == "cheapest-quote":
        steps = [
            call(0.4, "quotes.get", {"vendor": "Acme"}),
            result(1.0, "quotes.get", "Acme quote $89.00"),
            call(1.4, "quotes.get", {"vendor": "Beta"}),
            result(2.0, "quotes.get", "Beta quote $74.50"),
            call(2.4, "quotes.get", {"vendor": "Gamma"}),
            result(3.0, "quotes.get", "Gamma quote $92.25"),
            answer(3.4, "Cheapest is Beta at $74.50."),
        ]
    else:
        steps = [
            think(0.0, "I remember the policy is 14 days, pretty sure."),
            answer(0.4, "No — returns are only accepted within 14 days."),
        ]
    return dict(TASKS[task_id], task_id=task_id, steps=steps)


AGENTS = {"careful": careful, "sloppy": sloppy, "unlucky": unlucky}


def main():
    print("trace-judge demo — outcome-only scoring vs trace scoring\n")
    summary = {}
    for name, build in AGENTS.items():
        print(f"### agent: {name}")
        outcomes, overalls = 0, []
        for task_id in TASKS:
            report = evaluate(Trace(build(task_id)))
            print(render(report))
            print()
            outcomes += report["results"][0].score == 1.0
            overalls.append(report["overall"])
        summary[name] = (outcomes, sum(overalls) / len(overalls))

    print("### the comparison that matters")
    print(f"  {'agent':<10} {'outcomes':<9} {'trace score':<11}")
    for name, (outcomes, avg) in summary.items():
        print(f"  {name:<10} {outcomes}/3{'':<6} {avg:.2f}")
    print()
    print("Outcome-only scoring ties careful and sloppy at 3/3. Trace scoring")
    print("doesn't: sloppy repeats a call, never re-checks its math, and invents")
    print("a $4.99 shipping fee that appears nowhere — while unlucky sails past")
    print("a failed lookup and answers with a made-up total. That's the whole")
    print("point: the answer can be right while the run is still not one you'd ship.")


if __name__ == "__main__":
    main()
