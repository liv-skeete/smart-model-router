# SMR Scratchpad

This file is a temporary workspace for planning, drafting, and storing intermediate results during development. It can be safely cleared as needed.

**--- clear below here when starting new session ---**


# SMR Router – Implementation Plan (User-only classification + Qwen stability)

Goal
- Make routing strictly driven by user intent (exclude assistant/system).
- Reduce classifier payload size and variance to stabilize latency (especially for qwen2.5).
- Optionally isolate the router to a dedicated lightweight local model (qwen2.5:7b) with conservative options.

Key changes

1) Valves: add classifier controls
- File: [SMR_v1_Module.py]()
- In [python.class Filter.Valves(BaseModel)](SMR_v1_Module.py:45):
  - Add:
    - classifier_user_only: bool = True
    - classifier_max_user_messages: int = 3
    - classifier_max_chars_per_user_msg: int = 1024
    - classifier_max_transcript_chars: int = 4000
  - Keep existing:
    - history_window (for backward-compat/default path)
    - classifier_temperature (stay at 0.0)

2) Behavior: build transcript from user-only messages
- Function: [python.def _get_model_recommendation(...)](SMR_v1_Module.py:225)
- Replace transcript construction logic:
  - Current: take last N messages (all roles) [python.messages = body.get("messages", [])[-self.valves.history_window:]](SMR_v1_Module.py:244), then include user+assistant in transcript [python.for msg in messages:](SMR_v1_Module.py:246)
  - New:
    - If valves.classifier_user_only True:
      - Gather the last valves.classifier_max_user_messages messages with role == "user" from entire thread (not limited by history_window).
      - For each, flatten text and truncate to valves.classifier_max_chars_per_user_msg.
      - Join into transcript (most recent last).
      - After join, hard-cap the transcript string length to valves.classifier_max_transcript_chars.
    - Else:
      - Keep legacy behavior (use history_window slice and include roles).
  - Keep get_last_user_message() emphasis for “Final user request”.

3) Small prompt hygiene
- Ensure the transcript header says “Recent conversation (user-only…)” when user-only mode is on.
- Retain the “Final user request (focus primarily on this): …” block exactly as is.

4) Logging & diagnostics (temporary while validating)
- In [python._get_model_recommendation()](SMR_v1_Module.py:225):
  - Add a debug-only line to log the classifier payload size (len(json.dumps(payload))) behind valves.verbose_logging.
  - Continue logging elapsed_ms (already present).
  - Add which mode was used: user_only vs legacy.

5) Conservative defaults
- Set valves defaults:
  - classifier_user_only = True
  - classifier_max_user_messages = 3 (or 2 if you prefer even tighter)
  - classifier_max_chars_per_user_msg = 1024
  - classifier_max_transcript_chars = 4000
  - history_window = 1 if you want an immediate effect even before code changes get deployed
- Keep classifier_temperature=0.0

6) Router model isolation (optional but recommended)
- Create a dedicated local model for routing: qwen2.5:7b (Ollama).
- In model params (Open WebUI model configuration -> params -> “custom_params” that map via [python.apply_model_params_to_body_ollama()](Reference/open_webui/utils/payload.py:118)):
  - temperature: 0.0
  - num_ctx: 1024–2048
  - num_thread: 4–6
  - num_batch: 1–2
  - keep_alive: "5m"
  - do not enable think/mirostat/system
- Point SMR router to this entry:
  - [python.Filter.Valves.router_model_id](SMR_v1_Module.py:63) = "qwen2.5:7b" (or your configured model ID)
- If multiple Ollama nodes: pin router model to a single node to avoid cross-node variance (configure your model mapping accordingly).

7) Avoid classifier metadata bloat
- The dispatcher merges request.state.metadata into the request body for the classifier call: [python.generate_chat_completion() merge](Reference/open_webui/utils/chat.py:171-179)
- Upstream payloads should avoid large metadata (files/variables) before routing.
- If needed later, add an internal escape hatch to call the Ollama router endpoint directly with a minimal payload (but first try just keeping metadata lean).

Test plan

A) Functional
- Scenarios:
  - Single user question (“I am sad”) → model = fast (low latency)
  - Multi-turn with verbose assistant → last user short → model decided on user-only input
  - RAG or tool-using chats → router remains unaffected by assistant/tool verbosity
- Confirm classifier result correctness and stability.

B) Latency
- Measure “Classifier result: elapsed_ms=…” across:
  - history_window=1 (no code change baseline)
  - user-only mode (new behavior) with qwen2.5:14b
  - user-only mode with qwen2.5:7b dedicated router
  - gpt-4.1-nano (remote) baseline
- Targets:
  - Local qwen2.5:7b user-only: sub-300ms typical
  - Local qwen2.5:14b user-only: reduced variance vs before
  - Remote gpt-4.1-nano: ~1300ms (stable, but network bound)

C) Payload size
- With valves.verbose_logging=True capture:
  - transcript char counts (per line and total)
  - payload JSON size (bytes)
- Validate significant reduction vs current multi-role transcript.

Rollback/Safety
- Toggle classifier_user_only=False to revert to legacy behavior.
- Set history_window=1 as a simple, zero-risk mitigator even without code.
- Leave router_model_id as-is or point back to gpt-4.1-nano if needed.

Change list (surgical)
- [python.class Filter.Valves(BaseModel)](SMR_v1_Module.py:45): add four new valves.
- [python.def _get_model_recommendation(...)](SMR_v1_Module.py:225):
  - Replace transcript construction to user-only mode gated by new valves.
  - Add truncation guards.
  - Add optional payload size logging in verbose mode.
  - Update transcript header text conditional.
- No changes to MRS (memory) required.

Rationale
- Routing should be driven by user intent. Assistant/system turns are verbose, stylistic, and often contain non-routing details. Excluding them makes the classifier prompt short, legible, and unambiguous, and reduces model-server parse overhead that was causing latency spikes.

Future enhancement (optional)
- Add an LRU cache keyed by short hash of the final user request to fast-return the same route for repeated identical requests within a short TTL (e.g., 60s). Helpful for repetitive batch testing; disabled by default.
