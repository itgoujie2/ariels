"""Protocol Version Compatibility check.

A customer's agent doesn't just need to work when *we* call it — it needs to
keep working when called by whatever real counterparts exist in the wild,
which may be built on an older or newer A2A SDK generation than the customer
used. That's a different question from "did our probe get a sensible reply,"
so it's checked separately here, deterministically, with zero LLM judgment.

Two independent failure modes:

  1. Discovery-layer: can the agent card even be *parsed* by a client built
     against a specific SDK version's AgentCard model? Confirmed live: a
     v1.0-style card (no top-level `url`, only `supportedInterfaces`) makes
     a2a-sdk 0.3.26's `AgentCard.model_validate` raise `url: Field required`
     before a single message is ever sent.

  2. Message-layer: does the JSON-RPC endpoint actually *accept* requests
     encoded in each known dialect, regardless of what the card declares?
     Since app.a2a_wire already speaks every known dialect, this is just
     firing one harmless message per dialect at the endpoint and recording
     accept/reject — no second real agent required to prove the gap.

Neither check is a hard pass/fail: supporting only one dialect is a
legitimate choice, not a bug. It's reported as a flagged risk with a defined
blast radius, same tiering spirit as PLAN.md §7's Tier 1/Tier 2 reference
agents.
"""

from __future__ import annotations

import httpx

from app.a2a_wire import KNOWN_DIALECTS, PING_TEXT, check_streaming_accepted, classify_error_text, send_message


def _classify_error(e: Exception) -> str:
    """A 401/403 means the server understood the dialect fine and is gating
    on credentials we don't have -- a completely different finding from
    "method not found" (server genuinely doesn't speak this dialect).
    Confirmed live: api.sursatech.com declares v1.0 and requires a bearer
    token (self-registration, per its own card); reporting that as "does
    not support the latest A2A protocol version" would be a real, misleading
    error in the report, not just an imprecise one — it doesn't support
    *unauthenticated* v1.0 calls, and we have no way to tell whether the
    dialect itself works until we have credentials."""
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403):
        return "auth_required"
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 402:
        # Same principle as auth_required, different gate: the server
        # understood the dialect fine and is asking for payment (x402 has
        # come up repeatedly this session -- 2s.io, MERCURY, Agoragentic,
        # PostalForm, Torify...). Unlike ambiguous JSON-RPC custom codes,
        # HTTP 402 is a real, standardized status worth its own bucket
        # rather than folding into "other". Confirmed live: gpt55.558686.xyz
        # gates even a first message/send call behind payment.
        return "payment_required"
    if isinstance(e, RuntimeError):  # send_message's own raise for a JSON-RPC-level error
        # Delegates to a2a_wire.py's own classify_error_text() -- the same
        # substring checks used to live here directly, now generalized
        # (text-only, not isinstance-based) so every other prober/native-
        # client layer can reuse the identical classification on whatever
        # exception type *their own* client library happens to raise. Kept
        # as a thin RuntimeError-typed wrapper here since this function's
        # own callers (probe_dialect_acceptance) still need the isinstance
        # dispatch above for the httpx-level auth/payment cases.
        # default="unsupported": this function's own original behavior for
        # any JSON-RPC-level error that isn't one of the other specific
        # buckets, preserved exactly via classify_error_text's own default
        # override (see its docstring).
        return classify_error_text(str(e), default="unsupported")
    return "other"


async def probe_dialect_acceptance(
    httpx_client: httpx.AsyncClient, rpc_url: str, extra_headers: dict | None = None
) -> dict:
    results = {}
    for dialect in KNOWN_DIALECTS:
        try:
            await send_message(httpx_client, rpc_url, dialect, PING_TEXT, extra_headers=extra_headers)
            results[dialect] = {"accepted": True, "reason": "ok", "error": None}
        except Exception as e:
            results[dialect] = {"accepted": False, "reason": _classify_error(e), "error": str(e)}
    return results


def declared_versions(card: dict) -> list[str]:
    versions = set()
    if "protocolVersion" in card:
        versions.add(card["protocolVersion"])
    for iface in card.get("supportedInterfaces", []):
        if "protocolVersion" in iface:
            versions.add(iface["protocolVersion"])
    return sorted(versions)


def card_schema_risks(card: dict) -> list[dict]:
    risks = []
    has_flat_url = "url" in card
    has_supported_interfaces = bool(card.get("supportedInterfaces"))

    if not has_flat_url:
        risks.append(
            {
                "risk": "missing_top_level_url",
                "detail": (
                    "Card has no top-level `url` field (only "
                    + ("`supportedInterfaces`" if has_supported_interfaces else "no interface info at all")
                    + "). Clients built on strict v0.3-era AgentCard models "
                    "(e.g. a2a-sdk<=0.3.x's A2AClient) will fail to parse "
                    "this card at all — confirmed live: this exact shape "
                    "raises 'url: Field required' in a2a-sdk 0.3.26, before "
                    "any message is ever sent."
                ),
            }
        )

    if has_flat_url and not has_supported_interfaces:
        risks.append(
            {
                "risk": "no_supported_interfaces_declared",
                "detail": (
                    "Card only declares a flat `url`, no `supportedInterfaces` "
                    "list. Newer clients that key off `supportedInterfaces` to "
                    "pick a protocol binding/version may not be able to tell "
                    "what this agent actually speaks."
                ),
            }
        )

    return risks


async def probe(
    httpx_client: httpx.AsyncClient, card: dict, rpc_url: str, extra_headers: dict | None = None
) -> dict:
    dialect_results = await probe_dialect_acceptance(httpx_client, rpc_url, extra_headers)
    accepted = [d for d, r in dialect_results.items() if r["accepted"]]
    auth_required = [d for d, r in dialect_results.items() if r.get("reason") == "auth_required"]
    internal_error = [d for d, r in dialect_results.items() if r.get("reason") == "internal_error"]
    payment_required = [d for d, r in dialect_results.items() if r.get("reason") == "payment_required"]
    inconclusive = len(auth_required) + len(internal_error) + len(payment_required)

    result = {
        "declared_versions": declared_versions(card),
        "dialects_tested": dialect_results,
        "accepted_dialects": accepted,
        "auth_required_dialects": auth_required,
        "internal_error_dialects": internal_error,
        "payment_required_dialects": payment_required,
        # Only counts dialects we could actually confirm one way or the
        # other -- a dialect gated by auth or payment, or one whose call
        # blew up on the target's own backend (confirmed live:
        # neva.dt-agent.co.uk leaking an upstream API-key failure through a
        # JSON-RPC -32603), is neither confirmed-accepted nor confirmed-
        # unsupported, so it shouldn't inflate "no dialect at all works"
        # into a false failure, nor "supports it" into a false pass.
        "no_dialect_accepted": len(accepted) == 0 and inconclusive == 0,
        "single_dialect_only": len(accepted) == 1,
        "card_schema_risks": card_schema_risks(card),
    }

    # Only bother if the card actually claims streaming support -- no point
    # flagging "streaming didn't work" for an agent that never said it would.
    # `capabilities` is spec'd as an object, but found live (metavision.click)
    # as a bare list of capability-name strings instead -- .get() on that
    # crashes with AttributeError, so guard the shape rather than assume it.
    capabilities = card.get("capabilities")
    if isinstance(capabilities, dict) and capabilities.get("streaming"):
        # Best-effort dialect for the streaming probe: whichever dialect our
        # own message/send check just confirmed works, since streaming and
        # non-streaming calls on the same endpoint should speak the same
        # dialect in practice; falls back to the latest known one.
        streaming_dialect = accepted[0] if accepted else KNOWN_DIALECTS[-1]
        result["streaming"] = await check_streaming_accepted(
            httpx_client, rpc_url, streaming_dialect, extra_headers
        )

    return result
