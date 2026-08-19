---
name: Client interop finding
about: A real A2A client library behaving unexpectedly against a real agent
title: ""
labels: finding
assignees: ""
---

<!--
This is the most valuable kind of issue for this repo -- see FINDINGS.md
for the kind of write-up we're looking for and how existing findings are
documented. A finding is a complete contribution on its own, even with no
fix attached.
-->

**Which package** (e.g. `crewai-client-checks/`, `langgraph-prober/`):

**What happened**
<!-- What did the client do, concretely -- wrong method/header/role, a
parse failure, a silently-swallowed error, a spec-violating field name,
etc. -->

**Expected behavior**
<!-- What the A2A spec / a well-behaved client should do instead. -->

**Repro**

```
venv/bin/python run_parity.py <target-url>   # or run_prober.py for the golden probers
```

- Target agent URL (if it's still up / shareable):
- Client library + version:
- Relevant excerpt of `runs/latest.json` or raw output:

**Regression test**
<!-- Have you added (or can you add) a test in tests/ that reproduces this
against a fixture, per CONTRIBUTING.md's "what done looks like" section?
Not required to open the issue, but very welcome. -->
