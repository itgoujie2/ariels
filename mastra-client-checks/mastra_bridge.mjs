// Persistent bridge process using Mastra's own first-party A2A client
// (@mastra/core/a2a's A2AAgent). Spawned once per skill by
// mastra_prober.py and kept alive for the whole multi-turn probe,
// since A2AAgent.resumeGenerate() only works within the SAME process:
// its run state (context_id/task_id/waitingForInput per runId) lives
// in an in-memory Map internal to the A2AAgent instance, not something
// reconstructable across separate process invocations. Reads
// newline-delimited JSON commands from stdin, writes one JSON response
// per line to stdout.
//
// Commands:
//   {"cmd":"init","base_url":"..."}
//     -> {"ok":true,"name":"...","url":"..."} or {"ok":false,"error":"..."}
//   {"cmd":"generate","runId":"...","text":"..."}
//     -> {"ok":true,"task":{...}|null,"message":{...}|null,"resumePayload":{...}|null}
//        or {"ok":false,"error":"..."}
//   {"cmd":"resume","runId":"...","text":"..."}
//     -> same shape as "generate"

import { createInterface } from 'node:readline';
import { A2AAgent } from '@mastra/core/a2a';

let agent = null;

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

// MastraA2AError (thrown by @mastra/core/a2a's shared #request() method,
// used for both getAgentCard() and generate()/resumeGenerate()) carries
// {status, url} in its own `.data` field -- but its `.message` text
// alone ("Remote A2A request failed with status 404.") never embeds
// which URL actually failed, so a card-fetch 404 and an RPC-endpoint
// 404 are otherwise indistinguishable from the Python side. Confirmed
// live against p0stman.com (a legacy-well-known-path-only agent) and
// by reading @mastra/core's own compiled source
// (dist/chunk-O2CHMYVQ.js's MastraA2AError class, dist/a2a/index.js's
// #request()/#getBootstrap()) -- appending the URL when present lets
// mastra_prober.py's own explanation logic tell these apart precisely,
// rather than guessing from message text alone.
function _formatError(e) {
  const url = e?.data?.url;
  return url ? `${e.constructor.name}: ${e.message} (url: ${url})` : `${e.constructor.name}: ${e.message}`;
}

async function handle(cmd) {
  if (cmd.cmd === 'init') {
    try {
      agent = new A2AAgent({ url: cmd.base_url, timeoutMs: 30000 });
      const card = await agent.getAgentCard();
      send({ ok: true, name: card.name, url: card.url });
    } catch (e) {
      send({ ok: false, error: _formatError(e) });
    }
    return;
  }

  if (cmd.cmd === 'generate' || cmd.cmd === 'resume') {
    if (agent === null) {
      send({ ok: false, error: 'agent not initialized -- send an "init" command first' });
      return;
    }
    try {
      const result =
        cmd.cmd === 'generate'
          ? await agent.generate(cmd.text, { runId: cmd.runId })
          : await agent.resumeGenerate({ message: cmd.text }, { runId: cmd.runId });
      send({
        ok: true,
        task: result.task ?? null,
        message: result.message ?? null,
        resumePayload: result.resumePayload ?? null,
      });
    } catch (e) {
      send({ ok: false, error: _formatError(e) });
    }
    return;
  }

  send({ ok: false, error: `unknown cmd: ${cmd.cmd}` });
}

// Commands are processed strictly one-at-a-time (chained on this promise)
// so that a burst of buffered stdin lines can never produce
// out-of-order responses, even though the Python driver only ever
// sends the next command after reading the previous response.
let queue = Promise.resolve();

const rl = createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let cmd;
  try {
    cmd = JSON.parse(trimmed);
  } catch (e) {
    send({ ok: false, error: `invalid JSON command: ${e.message}` });
    return;
  }
  queue = queue.then(() => handle(cmd)).catch((e) => send({ ok: false, error: `unhandled: ${e.message}` }));
});
