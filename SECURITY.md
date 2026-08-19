# Security

## Scope

This repo is a set of 14 offline testing tools you run yourself against an
agent URL you control or have permission to test. There's no hosted
service, no server this project operates, and no account system here — so
most traditional web-app vulnerability classes (auth bypass, data exposure
across tenants, etc.) don't apply to this repo directly.

Real risk areas worth reporting privately rather than as a public issue:
- A crafted A2A response that causes one of the 14 clients to execute
  code, write outside `runs/`, or otherwise misbehave beyond a normal
  parse failure — these probers are designed to point at agents you don't
  fully trust, so any such gap defeats that purpose.
- Anything that could leak the `ANTHROPIC_API_KEY` from a user's `.env` to
  the target agent or a third party.
- A dependency-level vulnerability affecting one of the pinned packages
  that we should pin around, mirroring how `ag2-client-checks/`'s
  `requirements.txt` already documents a pin against a known-breaking
  release.

## Reporting

Email **itgoujie2@gmail.com** with a description and, if you have one, a
minimal reproduction. Please don't open a public issue for anything in the
list above until there's been a chance to look at it.

For anything else — a client library parsing quirk, a false pass/fail, a
crash against a well-behaved agent — a normal public issue is the right
place; see [CONTRIBUTING.md](CONTRIBUTING.md).
