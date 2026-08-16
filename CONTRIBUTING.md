# Contributing

Thanks for considering a contribution. This repo is 14 independent testing
engines against the A2A protocol — most contributions will touch exactly
one of them.

## Before you start

- **For anything beyond a small, obvious fix, open an issue first**,
  describing what you found and how you plan to fix it. This avoids
  duplicate work and lets us align on approach before you invest time in a
  PR that might need to go a different direction.
- **Bug reports are a completely valid contribution on their own.** If
  you've found a real client library behaving unexpectedly against a real
  agent, that's valuable even without a fix attached — see
  [FINDINGS.md](FINDINGS.md) for the kind of thing we're looking for and
  how existing findings are written up.

## Project layout

Each top-level directory (`langgraph-prober/`, `adk-prober/`,
`*-client-checks/`) is fully self-contained: its own `venv`, its own
`requirements.txt`, its own test suite. There's no shared build system —
working on one engine never requires touching another.

Four files are kept **byte-identical across all 14 packages**:
`app/a2a_wire.py`, `app/assertions.py`, `app/llm.py`,
`app/version_probe.py`, and `app/report_render.py`. If you fix something
in one copy, the same fix needs to be ported to the other 13 — this is
mechanical (`diff` two copies to confirm they still match before and
after) but easy to forget. A PR that touches one of these files should
touch all 14 copies identically, or explain why it doesn't.

## Setting up one engine

```bash
cd <engine-directory>
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # add ANTHROPIC_API_KEY if the tests need it
PYTHONPATH=. venv/bin/pytest tests/ -v
```

The three cross-language layers need one extra one-time step before their
own tests will pass:
- `mastra-client-checks/`: `npm install` inside that directory.
- `adkgo-client-checks/`: `cd bridge && go build -o adkgo_bridge .`
- `langchain4j-client-checks/`: `cd bridge && mvn package`

## What "done" looks like for a fix

- **The offline test suite passes**, including any new test you add for
  the specific bug you fixed.
- **If you found the issue by testing against a real agent, add a
  regression test for it** — a fixture built from (or closely modeling)
  the real response shape, plus an assertion that the fix actually
  catches it. The point is that nobody should have to re-discover the
  same gap by hitting a live endpoint a second time.
- **If you can, verify live** — run the engine's own CLI
  (`run_prober.py <url>` / `run_parity.py <url>`) against a real agent
  before and after your change and confirm the behavior actually changed
  the way you expect. Not always possible (some findings only reproduce
  against a specific real agent that may not stay up), but it's the
  strongest signal a fix actually works, not just that it satisfies a
  mock.

## Commit style

One logical change per commit — don't bundle an unrelated fix into the
same commit as a feature. Write commit messages that explain *why*, not
just *what* (the diff already shows what changed).

## Code style

No enforced linter config yet — match the surrounding code. A few
conventions worth following:
- Comments explain non-obvious *why* (a real constraint, a workaround for
  a specific library bug, something that would surprise a reader) — not
  *what* the code does, which well-named identifiers should already make
  clear.
- Prefer a small, targeted fix over a broader refactor in the same PR,
  even if the refactor is tempting.
