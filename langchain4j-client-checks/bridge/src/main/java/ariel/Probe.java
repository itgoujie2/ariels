package ariel;

// One-shot CLI probe using LangChain4j's own first-party A2A client
// (dev.langchain4j.agentic.a2a's DefaultA2AService/A2AClientBuilder,
// backed by the official org.a2aproject.sdk a2a-java-sdk-client). See
// langchain4j_prober.py's docstring for the full design rationale and
// the real findings this uncovered.
//
// Unlike the Mastra and ADK-Go layers, this does NOT need a persistent
// process holding session state across turns: LangChain4j's own
// generated client interface accepts @A2AContextId/@A2ATaskId as plain
// method parameters, so continuation state can be passed in and read
// back out explicitly on each call -- a fresh JVM per turn is enough,
// simpler than the other two cross-language layers' bridge processes.
//
// Usage: java -jar langchain4j-bridge.jar <baseUrl> <text> <contextId|null> <taskId|null>
// Always prints exactly one line of JSON to stdout:
//   {"ok": true, "task": {...raw Task JSON...}}
//   {"ok": false, "error": "..."}

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agentic.a2a.A2AContextId;
import dev.langchain4j.agentic.a2a.A2ATaskId;
import dev.langchain4j.agentic.a2a.DefaultA2AService;
import dev.langchain4j.agentic.internal.A2AClientBuilder;
import dev.langchain4j.agentic.internal.A2AService;
import org.a2aproject.sdk.spec.Task;

import java.util.HashMap;
import java.util.Map;

public class Probe {
    public interface RemoteAgent {
        Task chat(String message, @A2AContextId String contextId, @A2ATaskId String taskId);
    }

    public static void main(String[] args) {
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> out = new HashMap<>();
        try {
            String baseUrl = args[0];
            String text = args[1];
            String contextId = (args.length > 2 && !"null".equals(args[2])) ? args[2] : null;
            String taskId = (args.length > 3 && !"null".equals(args[3])) ? args[3] : null;

            A2AService service = new DefaultA2AService();
            A2AClientBuilder<RemoteAgent> builder = service.a2aBuilder(baseUrl, RemoteAgent.class);
            RemoteAgent agent = builder.build();
            Task task = agent.chat(text, contextId, taskId);

            out.put("ok", true);
            out.put("task", mapper.convertValue(task, Map.class));
        } catch (Throwable t) {
            out.put("ok", false);
            out.put("error", t.getClass().getName() + ": " + t.getMessage());
        }
        try {
            System.out.println(mapper.writeValueAsString(out));
        } catch (Exception e) {
            System.out.println("{\"ok\": false, \"error\": \"failed to serialize result: " + e.getMessage().replace("\"", "'") + "\"}");
        }
    }
}
