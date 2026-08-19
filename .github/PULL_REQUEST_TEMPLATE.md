**Which package(s):**

**What this changes and why:**

**Checklist** (see CONTRIBUTING.md for detail)
- [ ] Offline test suite passes (`PYTHONPATH=. venv/bin/pytest tests/ -v`)
- [ ] If this fixes a bug found against a real agent, a regression test is included
- [ ] If this touches one of the 5 byte-identical shared files
      (`app/a2a_wire.py`, `app/assertions.py`, `app/llm.py`,
      `app/version_probe.py`, `app/report_render.py`), the same change is
      applied to all 14 copies
- [ ] Verified live against a real agent where possible
