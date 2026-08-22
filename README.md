# Ariel — A2A interoperability probers

[![tests](https://github.com/itgoujie2/ariels/actions/workflows/tests.yml/badge.svg)](https://github.com/itgoujie2/ariels/actions/workflows/tests.yml)

Independent testing engines for the [Agent2Agent (A2A) protocol](https://github.com/a2aproject/A2A) — two hand-rolled "golden" probers that speak the wire protocol directly, plus a growing set of layers that each drive a real agent through a different framework's own first-party A2A client (CrewAI, LangChain4j, Mastra, ADK-Go, Strands, and more — see the table below for the current full list). Point any of them at a live A2A agent's base URL and see, concretely, whether that specific client can actually reach it, discover its skills, and hold a real multi-turn conversation — not just whether the protocol spec says it should.

## Why so many separate implementations?

Every A2A client library makes its own choices about card discovery, dialect negotiation, continuation, and error handling — and those choices genuinely differ in ways that change whether a real caller can talk to a real agent. Testing through one SDK tells you whether *that* SDK's abstraction works. Testing through many independently-built clients tells you what a real, diverse ecosystem of callers will actually experience.

**See [FINDINGS.md](FINDINGS.md) for what that's actually turned up** — real, live-confirmed gaps in a dozen client libraries, including several official first-party ones.

## Layout

| Directory | What it tests |
|---|---|
| `langgraph-prober/`, `adk-prober/` | Hand-rolled A2A wire client (no third-party SDK dependency) — the ground-truth layer |
| `raw-sdk-client-checks/` | Plain `a2a-sdk` usage (`create_client()`/`ClientFactory`) |
| `ag2-client-checks/`, `crewai-client-checks/`, `strands-client-checks/`, `fasta2a-client-checks/`, `agno-client-checks/`, `praisonai-client-checks/`, `agent-framework-client-checks/` | Each framework's own genuine first-party A2A client |
| `native-client-checks/` | Google ADK's official `RemoteA2aAgent` client |
| `mastra-client-checks/`, `adkgo-client-checks/`, `langchain4j-client-checks/` | Cross-language clients (TypeScript, Go, Java) via a small persistent bridge subprocess |

Each directory is self-contained: its own `venv`/dependency manifest, its own offline test suite, its own CLI.

## Running a check

Each package works the same way — pick any one directory from the table
above:

```bash
cd langgraph-prober   # or any other directory from the table above
python -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY -- used only to phrase probe/follow-up text, never for pass/fail judgment
venv/bin/python run_prober.py https://your-agent.example.com    # golden probers
venv/bin/python run_parity.py https://your-agent.example.com    # the 12 native-client layers
```

This prints a JSON result and writes it to `runs/latest.json` alongside a self-contained `runs/latest.html` report you can open directly in a browser — no server, no account, nothing else needed.

### Example output

Real run, real agent, `overall_pass: true` — all 17 of its declared skills reached a genuine multi-turn `completed` state with real content back (trimmed below; the full output tests every skill individually):

```
$ venv/bin/python run_prober.py https://agent.co-legal.be
{
  "target": "https://agent.co-legal.be",
  "agent_name": "colegal-public-assistant",
  "agent_description": "Co-Legal Public Assistant — informational Q&A about Belgian and Dutch private-client legal and fiscal topics...",
  "declared_dialect": "1.0",
  "tested_dialect": "1.0",
  "declared_skills": [
    "answer_legal_question", "be.ecli.lookup", "eu.eurlex.lookup",
    "iban.validate", "be.vies.validate", "..."
  ],
  "skills_tested": [
    {
      "skill_id": "iban.validate",
      "final_state": "completed",
      "transcript": [
        {
          "sent": "Is IBAN BE68 5390 0754 7034 geldig?",
          "state": "completed",
          "received": "IBAN BE68 5390 0754 7034 is structureel geldig: de lengte en het controlegetal zijn correct. Dit bevestigt niet dat de rekening bestaat, actief is of aan een bepaalde persoon toebehoort."
        }
      ],
      "checks": [
        { "check": "reached_stopping_point", "passed": true, "detail": "stopped at 'completed' after 1 turn(s)" },
        { "check": "content_present_on_completion", "passed": true, "detail": "0 artifact part(s), final_message present" }
      ],
      "pass": true
    }
    // ...16 more skills, same shape
  ],
  "overall_pass": true
}
```

(Replies came back in Dutch, since that's the language this particular agent responded in — the checks only care whether a real conversation reached a real stopping point with real content, not what language it's in.)

The Go and Java layers (`adkgo-client-checks/`, `langchain4j-client-checks/`) need their bridge compiled once first (`cd bridge && go build` / `mvn package`); the TypeScript layer (`mastra-client-checks/`) needs `npm install` in `mastra-client-checks/`.

## Want to test the same agent through all of these at once, with a unified comparison report, continuous monitoring, and AI-judged goal completion?

That's what the hosted Ariel product does — it orchestrates every one of these engines together, renders one report with a cross-client comparison matrix, and can watch an agent over time and alert on regressions. This repo is the open-source core those probers are built from; the hosted product is a separate, additive layer on top, not a requirement for using any of this directly.

## Contributing

Bug reports (a real client behaving unexpectedly against a real agent) are
as welcome as code. See [CONTRIBUTING.md](CONTRIBUTING.md) for how this
repo is laid out, how to run one engine's own test suite, and what we look
for in a regression test. Community interactions here follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Found a security-relevant issue?
See [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
