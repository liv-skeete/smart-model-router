# SMR Changelog

## v1.3.0 - 2025-10-14
- **Code Analysis**: Completed full code review against Open WebUI Module Refactoring Guide
- **Documentation**: Analysis findings available in [Code_Analysis_Report.md](Code_Analysis_Report.md)
- **Compliance**: Module is production-ready and exceeds refactoring guide requirements
- **Note**: Only minor header update was required; module already followed best practices

## v1.2.8 - 2025-08-19
- UX: Prefer user email over UUID in classifier metadata for downstream logs/UX.
- No additional calls; routing behavior and persistence unchanged.
- Implementation: `metadata.user_id` now resolves to the user's email when available in [python.Filter._get_model_recommendation()](SMR/SMR_v1_Module.py:266).
## v1.2.7 - 2025-08-18
- **Rename:** Route tiers renamed to `fast` / `chat` / `deep` (plus `vision`) for clearer semantics.
- **Defaults:** Updated `Valves.routed_models` default mapping:
  - `fast` → `gpt-4.1-mini`
  - `chat` → `gpt-5-chat`
  - `deep` → `gpt-5`
  - `vision` → `gpt-4.1-mini`
- **Fallback:** `fallback_route_key` now defaults to `"fast"`.
- **Prompt:** Classifier prompt updated to v2.0; MODEL GUIDE and examples now use `fast` / `chat` / `deep`.
- **Docs/Comments:** Generalized classifier wording and removed legacy naming.
- **Version:** Module bumped to 1.2.7 with dev note reflecting tier rename and clean cut-over.

## v1.2.6 - 2025-08-17
- **UX**: Fallback status message now shows the route key used (e.g., "nuanced") instead of "None" for better clarity.
- **Config**: Added validation constraints (`ge=0`, `le=2`) to `classifier_temperature` valve to prevent misconfiguration.

## v1.2.5 - 2025-08-17
- Code Cleanup (Stage 5)
  - Removed unused `_format_log_content` helper function.
  - Removed redundant `confidence = None` assignments in classifier logic.

## v1.2.4 - 2025-08-17
- Prompt & Logging Enhancements (Stage 4)
  - Refined routing prompt with explicit instructions to reduce model drift and parsing errors.
  - Hardened logger configuration to prevent interference with host application's logging setup.

## v1.2.3 - 2025-08-17
- API Consistency & Efficiency (Stage 3)
  - Standardized `metadata.router` shape across all paths:
    - Added `elapsed_ms: 0` and `classifier_model_id: None` to vision and forced routes for consistency with the classifier path.
  - Performance: Cached parsing of `routed_models` JSON:
    - Reuses cached mapping when `valves.routed_models` is unchanged.
    - Falls back to last good cache on parse errors.
    - Emits verbose logs indicating cache usage and refresh.
## v1.2.2 - 2025-08-17
- Reliability & Hardening (Stage 2)
  - Classifier JSON parsing hardened:
    - Strips common markdown code fences from classifier output before parsing.
    - Wraps JSON parsing in try/except; on failure, logs (verbose) and safely falls back.
  - Valves validation:
    - `router_system_prompt` must be non-empty.
    - `history_window` now `ge=0`.
    - `classifier_timeout_seconds` now `ge=1`.
  - Configurable fallback route:
    - Added `fallback_route_key` valve and used it in fallback selection logic.
  - Legacy request cleanup:
    - Always strip legacy `options` from the top-level body and from `metadata` to conform to OpenAI-compatible schema.
    - Note: deviates from the draft plan (which proposed a toggle). The toggle was removed per deployment policy requiring only directly useful behavior.

## v1.2.1 - 2025-08-17
- **Concurrency Fix:** Made the `Filter` instance stateless by removing instance variables (`__request__`, `__user__`, `__model__`). This prevents race conditions and request cross-talk in concurrent environments.
- **Fix:** Corrected the forced model override logic to properly distinguish between the `route_key` and the `target_model_id`, ensuring consistent metadata and UI status messages.
- **Resilience:** Hardened user resolution logic to prevent runtime errors by handling cases where the user object is either a `dict` or an object.

## v1.2.0 - 2025-08-17
- **Architectural Change:** Refactored the module to make the `router_system_prompt` a required valve, removing the embedded default. The admin-provided prompt is now the single source of truth for routing logic, aligning with the design patterns of other advanced modules (e.g., MRS).
- **Architectural Change:** The router now embeds the user's request directly into the system prompt (via a `{user_request}` placeholder) for the classifier model. This is a best-practice change that improves the reliability and predictability of the routing decision by reducing the risk of conversational instruction drift.
- **Feature:** Added three new valves for fine-tuning router performance and behavior:
  - `history_window`: Controls how many recent messages are included in the classifier's context.
  - `classifier_timeout_seconds`: Sets the maximum wait time for a routing decision.
  - `classifier_temperature`: Allows adjustment of the classifier's determinism (0.0 is recommended).
- **Enhancement:** Added the full system prompt sent to the classifier model to the `verbose_logging` output, enabling easier diagnostics and prompt tuning.
- **Fix:** Added a safety check to prevent errors if the `router_system_prompt` is empty at runtime.

## v1.1.1 - 2025-08-16
- **Fix:** Corrected a fallback logic error where an incorrect label was used.
- **Fix:** Corrected typos in the system prompt.
- **Enhancement:** Changed UI status message from "Selected: {model_id}" to "Mode: {label}" for clarity.

## v1.1.0 - 2025-08-16
- **Feature:** Implemented a new standalone logging system, mirroring the pattern from the MRS module.
- **Enhancement:** All logging is now at the `INFO` level, removing the need for system-wide `DEBUG` settings to see module output.
- **Enhancement:** Replaced the `debug` valve with a `verbose_logging` valve.
  - When `verbose_logging` is `False` (default), only high-level summaries of routing decisions are logged.
  - When `verbose_logging` is `True`, detailed, data-rich logs are emitted for deeper diagnostics.
- **Enhancement:** Added comprehensive logging to all key decision points, providing clear visibility into the router's behavior in both standard and verbose modes.