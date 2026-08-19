---
name: Bug report
about: Something in this repo itself is broken (not a client library's own behavior)
title: ""
labels: bug
assignees: ""
---

<!--
If what you found is a real A2A client library misbehaving against a real
agent, use the "Client interop finding" template instead -- that's a
different, and very welcome, kind of report.
-->

**Which package** (e.g. `crewai-client-checks/`, `langgraph-prober/`):

**What happened**

**What you expected**

**Steps to reproduce**

```
venv/bin/pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. venv/bin/pytest tests/ -v   # or the exact command that fails
```

**Environment**
- OS:
- Python version:
- Package version / commit:
