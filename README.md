# trace-judge

Yesterday (Nov 19) Stripe published Stripe Evals — a benchmark that scores coding agents on building real Stripe integrations. The headline number stuck with me: across 14 model/harness pairs, agents hit 78% of the required criteria on average, but only 29% of runs passed *everything*. The gap between "mostly right" and "actually right" is the whole story, and their benchmark still mostly scores the finished code. I kept thinking about the other half: the trace. The 60-odd turns an agent took to get there. Did it look things up or guess? Did it notice when a call failed? Did the numbers in its answer come from anywhere?

So I built the small version of that question. trace-judge takes an agent run — a JSON list of thinks, tool calls, tool results, and a final answer — and scores how the agent got there, not just where it ended up. Five deterministic judges, no model calls, no dependencies. The point isn't to replace outcome evals. It's to catch the runs where the answer is right and the run still isn't one you'd ship.

## What it does

```bash
python demo.py                              # the comparison: 3 agents, same 3 tasks
python trace_judge.py examples/good-trace.json examples/bad-trace.json
python -m unittest discover -s tests       # 11 tests
```

A trace is just JSON. Steps are `think`, `tool_call`, `tool_result`, or `answer`. The five judges:

- **outcome** — does the final answer contain the expected one?
- **grounding** — every number in the answer: does it appear in a tool result, in the task itself, or as a derivation the answer shows *and* that checks out? A total the agent computed but never showed its working for counts as unverified, which is the point — I can't tell a correct sum from an invented one if the working isn't there.
- **tool_use** — duplicate calls with identical args, and failed or empty results the agent sailed past without retrying or even acknowledging. A retry after a failure doesn't count against you; repeating a call that already succeeded does.
- **verification** — did the agent re-check anything before answering? A "let me verify" think step, or showing the derivation. Straight from the first results to the answer scores low.
- **economy** — calls, tokens, and estimated cost against the task's par, if one is given. The rates are placeholders ($3/1M in, $15/1M out) — swap in whatever you actually pay.

Overall is a weighted mean (outcome .30, grounding .25, tool_use .20, verification .15, economy .10). A trace FAILs outright if the answer is wrong or grounding drops below 0.5.

## What I got

The demo runs three agents on three small tasks (add up three invoices, pick the cheapest of three quotes, answer a refund-policy question):

```
agent      outcomes  trace score
careful    3/3       1.00
sloppy     3/3       0.71
unlucky    1/3       0.51
```

Outcome-only scoring ties careful and sloppy at 3/3. Trace scoring doesn't, and the reasons are the interesting part:

- Sloppy gets every answer right and still fails the invoice task outright (0.56): it repeats the INV-101 lookup it already did, never shows its addition, so the $486.20 total is unverified — the judge can't distinguish it from a guess. On the quotes task it invents a "$4.99 shipping" fee that appears in no tool result anywhere. Grounding catches both.
- Unlucky's invoice run is the silent-failure case: the INV-102 lookup times out, it never retries, and it answers $420.00 anyway — a number it made up. Outcome 0, grounding 0, and the tool_use judge names the exact step where it went wrong.
- Careful is boring in the best way: looks everything up, shows the math, re-checks, scores 1.00 on all three. The boring agent is the one you'd actually put in front of a customer.

What surprised me: how much signal there is in the cheap checks. I expected grounding to be the interesting judge and verification to be noise, but the verification judge is what separates careful from sloppy most cleanly — "did you stop and check before answering" turns out to be a real behavioral fingerprint. The other surprise was the retry logic: my first version flagged *any* repeat call, including a legitimate retry after a failure, which is exactly backwards. A retry is good behavior; a repeat of a success is waste. Telling them apart took a small similarity check on the args (so a typo'd search query, fixed and retried, counts as a retry — but looking up a *different* invoice after one failed does not).

## How the pieces fit

`trace_judge.py` is the whole library: a `Trace` class that wraps the JSON, five judge functions that each return a score plus findings, and a weighted overall with a verdict. `demo.py` builds the three agents as step-lists over the same tasks and prints the comparison table. `examples/` has one good and one bad trace you can run through the CLI directly. `tests/` covers the judges' edge cases — the ones I actually got wrong while building it, like the retry-vs-duplicate distinction.

## Rough edges

- The grounding check only understands numbers. If an agent invents a *fact* in words ("the policy says 14 days"), the outcome judge catches it only if the expected answer disagrees. Text grounding needs embeddings or an LLM judge, and that's a different project.
- The verification judge is a keyword heuristic ("check", "verify", "confirm" in a think step). An agent could game it by writing "let me verify" and then not verifying. I know. It's a demo, not a benchmark — the honest version of this judge reads the *next* tool call, not the word.
- The derivation check only handles addition (`a + b = c`). Subtraction, percentages, currency conversion — all unhandled, all flagged as unverified. That's a false positive waiting to happen on real traces.
- Numbers under 10 with no decimal and no `$` are skipped as too generic ("3 invoices"). Reasonable, but it means a hallucinated "7 days" slips through. The threshold is a constant at the top if you disagree with it.
- Economy needs a par per task to mean anything. Without one it just reports the cost and scores 1.0, which is honest but not useful for comparing agents.

## TODO

- A text-grounding judge — check quoted phrases from the answer against tool results, not just numbers.
- Make verification structural: a re-check should be a tool call that re-reads the source, not a keyword in a think step.
- Handle subtraction and percentages in derivations, and flag arithmetic that *doesn't* check out as its own finding instead of lumping it under grounding.
- An LLM-judge adapter with the same interface, so the deterministic judges and a model judge can score the same trace side by side. The deterministic ones stay the default — they're free, fast, and reproducible.
- Real traces. Everything here is simulated; the next honest step is pointing this at an actual agent run (my DocuChat traces are the obvious first target) and seeing which findings survive contact with reality.

## Why this week

Stripe Evals landing yesterday made agent evals the freshest story around, and its numbers are the argument for this repo in miniature: 78% average criteria, 29% strict pass. The industry is learning to measure agents by what they produce. The trace is the other half of that measurement, and it's small enough to build from scratch in an evening — five judges, no model calls, eleven tests. If you're shipping agents, the question isn't just "did it get the answer right." It's "would you trust the run that produced it." This is my first attempt at scoring that.
