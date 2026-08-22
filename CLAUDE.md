# Ariel probers

Open-source core of the Ariel project: independent A2A interoperability
testing engines — 2 hand-rolled "golden" probers (`langgraph-prober/`,
`adk-prober/`) that speak the A2A wire protocol directly, plus a growing
set of native-client layers (`*-client-checks/`) that each drive a real
agent through a different framework's own first-party A2A client. See
`README.md` for the current full list, what each one does, and how to run
it.

This repo is a **manually-synced mirror** of `reference-agents/ours/*`
inside a separate, private sibling repo (`ariels-private`, at
`../ariels-private` on this machine as of 2026-08) that also contains the
hosted product built on top of these same probers. There is no
submodule/subtree link between them — keeping them in sync is a manual
discipline, not automated.

## Workflow

- **Whenever you change a file inside one of the package directories
  here, apply the identical change to the matching file in
  `../ariels-private/reference-agents/ours/<same package>/<same relative
  path>`, automatically, as part of the same turn — don't wait to be
  asked.** The private repo's hosted-product orchestrator runs against
  that copy directly, so the two must never silently drift apart. This
  applies in both directions: a change made while working in
  `ariels-private` under `reference-agents/ours/*` should be mirrored back
  out to this repo too.
  - This only covers the package directories themselves. This repo's
    own root-level files (`README.md`, `LICENSE`, `.gitignore`,
    `CLAUDE.md`) have no private-repo counterpart and should never be
    copied anywhere.
  - Committing the change in both repos is part of this same automatic
    habit. **Pushing to either remote is not** — this repo is public, so
    still confirm explicitly before pushing here, same standing practice
    as any other externally-visible action.
- Whenever probing a real agent turns up an issue — a new response shape,
  a parsing gap, a version/dialect quirk, a behavioral/conformance
  finding — add a regression test for it in the relevant package's
  `tests/` before moving on, same discipline as `ariels-private`'s own
  standing rule. Since `a2a_wire.py`/`assertions.py`/`llm.py`/
  `version_probe.py`/`report_render.py` are kept byte-identical across
  every package, a fix in one should be ported to every other copy too,
  and its regression test copied alongside it.
