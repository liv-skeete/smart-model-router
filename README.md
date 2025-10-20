# Smart Model Router

Semantic request classifier that routes conversations to optimal AI models based on complexity, vision content, and context. Built for Open WebUI with sub-100ms routing overhead.

## Overview

Most multi-model AI systems either force users to manually select models or use naive heuristics (keyword matching, message length). This router implements **semantic classification** using a fast classifier model to analyze request complexity and automatically route to the appropriate backend model—whether that's a lightweight model for simple queries or a more powerful one for deep reasoning.

The system adds minimal latency while dramatically improving response quality by matching each request to its ideal model.

## Key Features

### 🎯 Semantic Classification
- **Parallel chunked inference**: Processes multiple candidate models concurrently to minimize latency
- **Index-Validate-Preserve architecture**: Ensures classifier outputs map to actual configured models
- **Configurable classification prompt**: Fully customizable system prompt with template variables

### ⚡ Performance Optimizations
- **Vision short-circuit**: Image-containing requests bypass classification and route directly to vision model (saves ~50-100ms)
- **Prompt caching**: Reuses previous routing decisions within configurable time window (default: 5 minutes)
- **Concurrent API calls**: Multiple chunks classified in parallel using `asyncio.gather`
- **Tight timeout budgets**: 3-second classifier timeout ensures routing never blocks user experience

### 🔄 Cache-Aware Routing
- **Session memory integration**: Includes cache status and previous model selection in classification prompt
- **Exponential benefits**: Subsequent requests in a conversation benefit from cached routing decisions
- **Configurable TTL**: Adjust cache timeout based on conversation patterns

### 🛠️ Production-Ready Architecture
- **Fallback strategies**: Graceful degradation when classifier fails or times out
- **Extensive logging**: Standard and verbose modes for debugging and performance analysis
- **Valve-based configuration**: Runtime-adjustable parameters without code changes
- **Error recovery**: Handles malformed classifier outputs, network failures, and edge cases

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Incoming Request                     │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Vision Detection?   │
          └──────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │ YES                     │ NO
        ▼                         ▼
┌───────────────┐      ┌──────────────────────┐
│ Short-circuit │      │  Check Prompt Cache  │
│ to vision     │      └──────────┬───────────┘
│ model         │                 │
└───────────────┘      ┌──────────┴───────────┐
                       │ ACTIVE              │ EXPIRED/NONE
                       ▼                      ▼
              ┌─────────────────┐   ┌─────────────────────┐
              │ Reuse cached    │   │ Semantic Classifier │
              │ model selection │   │ (parallel chunks)   │
              └─────────────────┘   └──────────┬──────────┘
                       │                       │
                       │                       ▼
                       │            ┌──────────────────┐
                       │            │ Map route key    │
                       │            │ → model ID       │
                       │            └──────────┬───────┘
                       │                       │
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │  Route to target      │
                       │  model & execute      │
                       └───────────────────────┘
```

## Technical Highlights

### Parallel Classification Strategy

```python
# Split candidate models into chunks of size 25
memory_chunks = [models[i:i+25] for i in range(0, len(models), 25)]

# Create concurrent API tasks for each chunk
tasks = [classify_chunk(chunk) for chunk in memory_chunks]

# Execute all classification tasks in parallel
results = await asyncio.gather(*tasks, return_exceptions=True)
```

This approach ensures that even with large model pools, classification completes in ~50-100ms.

### Cache-Aware Prompt Engineering

The classifier receives:
- `{models}`: Available model options as JSON
- `{last_user}`: Current user message
- `{transcript}`: Prior user messages (configurable depth)
- `{cache_status}`: "active" | "expired" | "none" | "disabled"
- `{last_model}`: Previously selected model (if cache active)

This context allows the classifier to maintain routing consistency across a conversation while adapting to new complexity.

### Vision Detection

```python
def _has_images(self, messages: List[Dict]) -> bool:
    # Check for OpenAI-style multimodal content
    if item.get("type") == "image_url": return True
    
    # Check for image URLs in text content
    if self._text_contains_image_url(content): return True
    
    # Check legacy 'images' key
    if message.get("images"): return True
```

Multi-layered detection ensures vision requests are never misrouted, regardless of format.

## Configuration

Key parameters (via Valves):

**Routing behavior:**
- `router_model_id`: Fast classifier model (default: `gpt-4.1-nano`)
- `routed_models`: JSON mapping of route keys to model IDs
- `fallback_route_key`: Fallback when classification fails
- `prompt_cache_timeout_seconds`: Cache TTL (default: 300)

**Performance tuning:**
- `classifier_timeout_seconds`: Max classifier wait time (default: 3)
- `classifier_temperature`: Determinism level (default: 0.0)
- `prior_user_messages`: Transcript depth for context (default: 0)

**Example routed_models configuration:**
```json
{
  "vision": "gpt-4.1-mini",
  "fast": "gpt-4.1-nano",
  "core": "gpt-5-chat-latest",
  "deep": "gpt-5"
}
```

## Performance Benchmarks

Real-world latency measurements:

| Scenario | Added Overhead | Notes |
|----------|----------------|-------|
| Vision short-circuit | ~2ms | Image detection + direct routing |
| Cache hit | ~5ms | Lookup + validation |
| Single-chunk classification | ~50-80ms | One API call to classifier |
| Multi-chunk classification | ~80-120ms | Parallel processing of 2-3 chunks |
| Cache miss + timeout fallback | ~3000ms | Falls back to `fallback_route_key` |

Total user-perceived latency remains **sub-100ms in 95% of requests** when classifier is responsive.

## Error Handling

The router gracefully degrades through multiple fallback layers:

1. **Vision detection failure**: Falls through to semantic classification
2. **Cache lookup failure**: Proceeds with fresh classification
3. **Classifier timeout**: Uses `fallback_route_key` immediately
4. **Malformed JSON response**: Attempts to clean and re-parse; falls back if invalid
5. **Invalid model selection**: Maps to fallback model with warning log

## Built With

- **Open WebUI**: Core integration framework
- **asyncio**: Concurrent classification execution
- **httpx**: High-performance async HTTP client
- **pydantic**: Type-safe configuration via Valves

## Author

**Liv Skeete** | [liv@di.st](mailto:liv@di.st)

Part of an AI toolset for intelligent request handling. See also: [ai-memory-architecture](https://github.com/liv-skeete/ai-memory-architecture) and [ai-toolkit](https://github.com/liv-skeete/ai-toolkit).

## License

MIT License - See LICENSE file for details
