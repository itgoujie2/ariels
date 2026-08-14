// Persistent bridge process using Google ADK for Go's own first-party
// A2A client (google.golang.org/adk/agent/remoteagent/v2's NewA2A).
// Spawned once per skill by adkgo_prober.py and kept alive for the
// whole multi-turn probe, since the runner+session state (context_id/
// task_id tracked internally by ADK's own session service) lives in
// this process's memory, not something reconstructable across
// separate process invocations -- the same reason the Mastra layer's
// bridge is also a persistent process. Reads newline-delimited JSON
// commands from stdin, writes one JSON response per line to stdout.
//
// Commands:
//
//	{"cmd":"init","base_url":"..."}
//	  -> {"ok":true} or {"ok":false,"error":"..."}
//	{"cmd":"generate","sessionId":"...","text":"..."}
//	{"cmd":"resume","sessionId":"...","text":"..."}
//	  (both aliased to the same call: a plain new message on the same
//	  session -- see adkgo_prober.py's docstring for why there is no
//	  distinct, working continuation call for a plain-text
//	  input-required response with this client)
//	  -> {"ok":true,"response":{...}|null,"taskId":"...","contextId":"..."}
//	     or {"ok":false,"error":"..."}
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"

	"google.golang.org/adk/agent"
	remoteagent "google.golang.org/adk/agent/remoteagent/v2"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type command struct {
	Cmd       string `json:"cmd"`
	BaseURL   string `json:"base_url"`
	SessionID string `json:"sessionId"`
	Text      string `json:"text"`
}

type response struct {
	OK        bool   `json:"ok"`
	Error     string `json:"error,omitempty"`
	Response  any    `json:"response,omitempty"`
	TaskID    string `json:"taskId,omitempty"`
	ContextID string `json:"contextId,omitempty"`
}

var (
	theRunner *runner.Runner
)

func send(w *bufio.Writer, r response) {
	b, _ := json.Marshal(r)
	w.Write(b)
	w.WriteByte('\n')
	w.Flush()
}

func handleInit(cmd command) response {
	remoteAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:              "probe_agent",
		Description:       "probe agent for Ariel's native-client testing",
		AgentCardProvider: remoteagent.NewAgentCardProvider(cmd.BaseURL),
	})
	if err != nil {
		return response{OK: false, Error: fmt.Sprintf("%T: %v", err, err)}
	}
	sessSvc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "ariel-adkgo-prober",
		Agent:             remoteAgent,
		SessionService:    sessSvc,
		AutoCreateSession: true,
	})
	if err != nil {
		return response{OK: false, Error: fmt.Sprintf("%T: %v", err, err)}
	}
	theRunner = r
	return response{OK: true}
}

func handleTurn(ctx context.Context, cmd command) response {
	if theRunner == nil {
		return response{OK: false, Error: "runner not initialized -- send an \"init\" command first"}
	}
	msg := genai.NewContentFromText(cmd.Text, genai.RoleUser)
	var lastResp any
	var taskID, contextID string
	for event, err := range theRunner.Run(ctx, "ariel-user", cmd.SessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return response{OK: false, Error: fmt.Sprintf("%T: %v", err, err)}
		}
		if event.CustomMetadata == nil {
			continue
		}
		if errMsg, ok := event.CustomMetadata["a2a:error"]; ok {
			return response{OK: false, Error: fmt.Sprintf("%v", errMsg)}
		}
		if resp, ok := event.CustomMetadata["a2a:response"]; ok {
			lastResp = resp
		}
		if tid, ok := event.CustomMetadata["a2a:task_id"]; ok {
			taskID = fmt.Sprintf("%v", tid)
		}
		if cid, ok := event.CustomMetadata["a2a:context_id"]; ok {
			contextID = fmt.Sprintf("%v", cid)
		}
	}
	if lastResp == nil {
		return response{OK: false, Error: "adk-go client returned no a2a:response in any event"}
	}
	return response{OK: true, Response: lastResp, TaskID: taskID, ContextID: contextID}
}

func main() {
	ctx := context.Background()
	reader := bufio.NewReader(os.Stdin)
	writer := bufio.NewWriter(os.Stdout)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return
		}
		if len(line) == 0 {
			continue
		}
		var cmd command
		if err := json.Unmarshal([]byte(line), &cmd); err != nil {
			send(writer, response{OK: false, Error: fmt.Sprintf("invalid JSON command: %v", err)})
			continue
		}
		switch cmd.Cmd {
		case "init":
			send(writer, handleInit(cmd))
		case "generate", "resume":
			send(writer, handleTurn(ctx, cmd))
		default:
			send(writer, response{OK: false, Error: fmt.Sprintf("unknown cmd: %q", cmd.Cmd)})
		}
	}
}
