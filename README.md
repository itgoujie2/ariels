# Ariel — A2A interoperability probers

Fourteen independent testing engines for the [Agent2Agent (A2A) protocol](https://github.com/a2aproject/A2A) — two hand-rolled "golden" probers that speak the wire protocol directly, plus twelve layers that each drive a real agent through a different framework's own first-party A2A client (CrewAI, LangChain4j, Mastra, ADK-Go, Strands, and more). Point any of them at a live A2A agent's base URL and see, concretely, whether that specific client can actually reach it, discover its skills, and hold a real multi-turn conversation — not just whether the protocol spec says it should.

## Why fourteen separate implementations?

Every A2A client library makes its own choices about card discovery, dialect negotiation, continuation, and error handling — and those choices genuinely differ in ways that change whether a real caller can talk to a real agent. Testing through one SDK tells you whether *that* SDK's abstraction works. Testing through fourteen independently-built clients tells you what a real, diverse ecosystem of callers will actually experience.

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

Each package works the same way. From inside any one directory:

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY -- used only to phrase probe/follow-up text, never for pass/fail judgment
venv/bin/python run_prober.py https://your-agent.example.com    # golden probers
venv/bin/python run_parity.py https://your-agent.example.com    # the 12 native-client layers
```

This prints a JSON result and writes it to `runs/latest.json` alongside a self-contained `runs/latest.html` report you can open directly in a browser — no server, no account, nothing else needed.

The Go and Java layers (`adkgo-client-checks/`, `langchain4j-client-checks/`) need their bridge compiled once first (`cd bridge && go build` / `mvn package`); the TypeScript layer (`mastra-client-checks/`) needs `npm install` in `mastra-client-checks/`.

## Want to test the same agent through all fourteen at once, with a unified comparison report, continuous monitoring, and AI-judged goal completion?

That's what the hosted Ariel product does — it orchestrates all fourteen of these engines together, renders one report with a cross-client comparison matrix, and can watch an agent over time and alert on regressions. This repo is the open-source core those probers are built from; the hosted product is a separate, additive layer on top, not a requirement for using any of this directly.

## Contributing

Bug reports (a real client behaving unexpectedly against a real agent) are
as welcome as code. See [CONTRIBUTING.md](CONTRIBUTING.md) for how this
repo is laid out, how to run one engine's own test suite, and what we look
for in a regression test.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
