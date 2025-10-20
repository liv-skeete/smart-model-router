"""
title: Smart Model Router
description: A minimal router for Open WebUI that selects a target model per request.
author: Cody
version: 1.3.0
date: 2025-08-19
changelog: _changelog.md
"""

import logging
import json
import re
import asyncio
import time
from typing import Callable, Awaitable, Any, Optional, List, Dict

from pydantic import BaseModel, Field
from fastapi import Request

from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message
from open_webui.models.users import Users

name = "semantic_router"

# Standalone logger (independent of system logger), INFO level with explicit handler
logger = logging.getLogger("model_router")
logger.propagate = False
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


class Filter:
    """
    Minimal Smart Model Router:
    - Vision short-circuit: route any image-containing request directly to routed_models["vision"].
    - Non-vision: classify to a model using the router model (router_model_id), then map route key -> model ID via routed_models.
    - Do not mutate tools/filters/files or deep model metadata; only change body["model"] and add a small metadata.router breadcrumb.
    """

    class Valves(BaseModel):
        # Set processing priority
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations",
        )
        status: bool = Field(
            True,
            description="Show status updates",
        )
        show_reasoning: bool = Field(
            False,
            description="Show routing reasoning in chat",
        )
        verbose_logging: bool = Field(
            False, description="Enable detailed diagnostic logging"
        )

        router_model_id: str = Field(
            default="gpt-4.1-nano",
            description="Model ID used ONLY to classify (routing decisions). Should be fast.",
        )
        routed_models: str = Field(
            default=(
                "{\n"
                '  "vision": "gpt-4.1-mini",\n'
                '  "fast": "gpt-4.1-nano",\n'
                '  "core": "gpt-5-chat-latest",\n'
                '  "deep": "gpt-5"\n'
                "}"
            ),
            description="JSON object mapping route keys -> model IDs. Example:",
        )
        fallback_route_key: str = Field(
            default="fast", description="Route key to use when classification fails."
        )
        router_system_prompt: str = Field(
            ...,  # Required; validated to be non-empty
            description="Classifier system prompt template. Must contain {models} and may use {last_user} and {transcript}. Assistant content is never included.",
        )
        prior_user_messages: int = Field(
            default=0,
            ge=0,
            description="Number of prior user messages to include in transcript (bulleted). last_user is always provided separately.",
        )
        classifier_timeout_seconds: int = Field(
            default=3,
            ge=1,
            description="Timeout in seconds for the classifier model's response.",
        )
        classifier_temperature: float = Field(
            default=0.0,
            ge=0.0,
            le=2.0,
            description="Temperature for the classifier model. 0.0 is recommended for deterministic routing.",
        )

    def __init__(self):
        try:
            self.valves = self.Valves(
                router_system_prompt=""  # Initialize with empty string; admin must provide a value.
            )
        except Exception as e:
            logger.error(f"Failed to initialize SMR valves: {e}")
            raise
        self._log_message("SMR initialized")

    def _log_message(
        self, standard_msg: Optional[str], verbose_msg: Optional[str] = None
    ) -> None:
        """
        Unified logger:
        - Always logs standard_msg at INFO.
        - Logs verbose_msg at INFO only when valves.verbose_logging is enabled.
        """
        if standard_msg:
            logger.info(standard_msg)
        if getattr(self.valves, "verbose_logging", False) and verbose_msg:
            logger.info(verbose_msg)

    @staticmethod
    def _truncate_log_lines(text: str, max_lines: int = 1000) -> str:
        """
        Truncate a multi-line string to at most max_lines for safe verbose logging.
        """
        lines = text.split("\n")
        if len(lines) <= max_lines:
            return text
        truncated = lines[:max_lines]
        omitted = len(lines) - max_lines
        truncated.append(f"... [truncated, {omitted} more lines omitted]")
        return "\n".join(truncated)

    @staticmethod
    def _text_contains_image_url(text: str) -> bool:
        """Return True if the text contains a likely image URL."""
        if not isinstance(text, str) or not text.strip():
            return False
        # Simplified regex for common image file extensions.
        # This is a lightweight check and may have false positives/negatives.
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"]
        # Check for http/https protocol and one of the extensions, case-insensitive.
        pattern = re.compile(
            r"https?://[^\s/$.?#].[^\s]*(" + "|".join(re.escape(ext) for ext in image_extensions) + r")",
            re.IGNORECASE,
        )
        return bool(pattern.search(text))

    def _has_images(self, messages: List[Dict]) -> bool:
        """Return True if the last user message contains images."""
        last_user_message_dict = None
        for message in reversed(messages or []):
            if message.get("role") == "user":
                last_user_message_dict = message
                break
        if not last_user_message_dict:
            return False

        content = last_user_message_dict.get("content")
        if isinstance(content, list):
            # OpenAI-style multimodal content list
            has_image_url_item = any(
                (isinstance(item, dict) and item.get("type") == "image_url")
                for item in content
            )
            if has_image_url_item:
                return True
            
            # Also check for text items that contain image URLs
            text_content = self._flatten_text(content)
            if self._text_contains_image_url(text_content):
                return True

        # Check for image URLs in simple string content
        elif isinstance(content, str):
            if self._text_contains_image_url(content):
                return True

        # Back-compat: some sources may use 'images' key
        return bool(last_user_message_dict.get("images"))

    def _load_routed_models(self) -> Dict[str, str]:
        """Load and validate routed_models mapping from valves with caching."""
        src = self.valves.routed_models or ""
        # Cache hit
        if getattr(self, "_routed_models_src", None) == src and isinstance(
            getattr(self, "_cached_routed_map", None), dict
        ):
            self._log_message(None, verbose_msg="Using cached routed_models mapping")
            return self._cached_routed_map or {}
        # Cache miss: parse
        try:
            data = json.loads(src or "{}")
            if not isinstance(data, dict):
                logger.warning("routed_models is not a JSON object; got %s", type(data))
                cleaned = {}
            else:
                # Keep only string -> string pairs
                cleaned = {
                    str(k): str(v)
                    for k, v in data.items()
                    if isinstance(k, str)
                    and isinstance(v, str)
                    and k.strip()
                    and v.strip()
                }
                if not cleaned:
                    logger.warning("routed_models mapping is empty after validation")
            # Update cache on successful parse
            self._cached_routed_map = cleaned
            self._routed_models_src = src
            self._log_message(
                None,
                verbose_msg=f"Parsed and cached routed_models ({len(cleaned)} route(s))",
            )
            return cleaned
        except Exception as e:
            logger.error("Failed to parse routed_models JSON: %s", str(e))
            # If we had a previous valid cache, prefer it over empty mapping
            if (
                isinstance(getattr(self, "_cached_routed_map", None), dict)
                and getattr(self, "_routed_models_src", None) is not None
            ):
                self._log_message(
                    None,
                    verbose_msg="Using previous cached routed_models due to parse error",
                )
                return self._cached_routed_map or {}
            return {}

    @staticmethod
    def _flatten_text(content: Any) -> str:
        """Extract text-only from message content. Supports string or OpenAI-style content lists."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(text, dict):
                        # Defensive: some SDKs may nest { type: "text", text: { ... } }
                        parts.append(json.dumps(text, ensure_ascii=False))
            return " ".join(p for p in parts if p).strip()
        return ""

    async def _get_model_recommendation(
        self,
        body: dict,
        models: List[str],
        request: Request,
        user: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Call the local router model to classify into one of the provided models.
        Returns: { "model": str, "reason": str|None, "elapsed_ms": int }
        On error/timeout: None
        """
        if (
            not self.valves.router_system_prompt
            or not self.valves.router_system_prompt.strip()
        ):
            logger.warning("Router system prompt is empty; skipping classification.")
            return None

        # Build user-only transcript: previous (N-1) user messages as bullets; last_user provided separately
        messages = body.get("messages", []) or []
        user_texts: List[str] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = self._flatten_text(msg.get("content"))
            if text:
                user_texts.append(text)

        last_user = get_last_user_message(messages or []) or ""

        prior_count = max(int(self.valves.prior_user_messages or 0), 0)
        prior_user_texts: List[str] = []
        if prior_count > 0 and len(user_texts) > 1:
            start_idx = max(len(user_texts) - 1 - prior_count, 0)
            prior_user_texts = user_texts[start_idx : len(user_texts) - 1][::-1]

        transcript_bullets = "\n".join(f"- {t}" for t in prior_user_texts)

        models_json = json.dumps(models, ensure_ascii=False)

        system_prompt = self.valves.router_system_prompt.format(
            models=models_json,
            last_user=last_user,
            transcript=transcript_bullets,
        )

        payload = {
            "model": self.valves.router_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Return JSON only."},
            ],
            "stream": False,
            "temperature": self.valves.classifier_temperature,
            # Prefer email in metadata to avoid exposing UUIDs in downstream logs/UX.
            "metadata": {"user_id": (getattr(user, "email", None) if user else None)},
        }

        self._log_message(
            f"Classifying with '{self.valves.router_model_id}'",
            verbose_msg=f"Classifier System Prompt:\n{self._truncate_log_lines(system_prompt)}",
        )

        start = time.perf_counter()
        try:
            # Tight budget to ensure low added latency
            response = await asyncio.wait_for(
                generate_chat_completion(
                    request, payload, user=user, bypass_filter=True
                ),
                timeout=self.valves.classifier_timeout_seconds,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            raw = response["choices"][0]["message"]["content"].strip()

            # Strip common markdown code fences like ```json ... ``` or ``` ... ```
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                # Split off the leading fence line, which may be ```json or ```
                if "\n" in cleaned:
                    first_line, rest = cleaned.split("\n", 1)
                    # Remove trailing closing fence if present
                    end_idx = rest.rfind("```")
                    if end_idx != -1:
                        cleaned = rest[:end_idx].strip()
                    else:
                        cleaned = rest.strip()
                else:
                    cleaned = cleaned.strip("`").strip()

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # On malformed JSON, log (verbose) and trigger fallback by returning None
                self._log_message(
                    None,
                    verbose_msg=f"Malformed classifier output (not JSON). Raw:\n{self._truncate_log_lines(raw)}",
                )
                return None

            model = data.get("model")
            reason = data.get("reason")

            if not isinstance(model, str) or model not in models:
                self._log_message(
                    None,
                    verbose_msg=f"Classifier returned invalid or unknown model: {model!r}",
                )
                return {
                    "model": None,
                    "reason": reason,
                    "elapsed_ms": elapsed_ms,
                }
            # Type hygiene
            if reason is not None and not isinstance(reason, str):
                reason = None

            self._log_message(
                f"Classifier result: model='{model}', elapsed_ms={elapsed_ms}",
                verbose_msg=f"reason={reason}",
            )

            return {
                "model": model,
                "reason": reason,
                "elapsed_ms": elapsed_ms,
            }
        except asyncio.TimeoutError:
            self._log_message(
                "Classifier timed out", verbose_msg="Router classification timed out"
            )
            return None
        except Exception as e:
            logger.error(
                "Error during router classification: %s", str(e), exc_info=True
            )
            return None

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> dict:

        user_id = None
        if isinstance(__user__, dict):
            user_id = __user__.get("id")
        elif hasattr(__user__, "id"):
            user_id = __user__.id

        user = Users.get_user_by_id(user_id) if user_id else None
        self._log_message(
            "Routing request started",
            verbose_msg=f"Incoming model: {body.get('model')}",
        )

        # Remove unsupported/legacy options at top level to conform to OpenAI-compatible schema
        body.pop("options", None)

        # --- Metadata safety net (coerce to dict) ---
        metadata = body.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(
                    "Metadata was a non-JSON string. Defaulting to empty dict."
                )
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        body["metadata"] = metadata

        # Remove options nested under metadata if present to conform to OpenAI-compatible schema
        body["metadata"].pop("options", None)

        # -------------------------------------------

        # Router always active (whitelist removed)

        # --- Per-request overrides (low-complexity utility) ---
        router_overrides = {}
        try:
            router_overrides = body["metadata"].get("router", {}) or {}
        except Exception:
            router_overrides = {}
        if (
            isinstance(router_overrides, dict)
            and router_overrides.get("disable") is True
        ):
            self._log_message(
                None, verbose_msg="Routing disabled via metadata.router.disable"
            )
            return body
        # ------------------------------------------------------

        # --- Load routed_models mapping ---
        routed_map = self._load_routed_models()
        if not routed_map:
            logger.warning("No routed_models configured; returning body unchanged.")
            return body
        self._log_message(
            f"Loaded {len(routed_map)} route(s).",
            verbose_msg=f"Routes: {', '.join(sorted(routed_map.keys()))}",
        )
        # ----------------------------------

        messages = body.get("messages", []) or []

        # --- Vision short-circuit ---
        if self._has_images(messages):
            self._log_message("Vision content detected")
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "👀 Routing to vision model",
                        },
                    }
                )
            vision_model = routed_map.get("vision")
            if not vision_model:
                logger.warning(
                    "Vision detected but 'vision' route is not configured in routed_models."
                )
                return body
            self._log_message(f"Vision short-circuit -> '{vision_model}'")

            # Breadcrumb
            body.setdefault("metadata", {})
            body["metadata"]["router"] = {
                "model": "vision",
                "model_id": vision_model,
                "reason": "short_circuit",
                "classifier_model_id": None,
                "elapsed_ms": 0,
            }
            if vision_model == body.get("model"):
                self._log_message("Vision model equals incoming; returning unchanged")
                return body

            body["model"] = vision_model
            return body
        # ---------------------------

        # --- Forced model override (no classifier) ---
        if isinstance(router_overrides, dict) and isinstance(
            router_overrides.get("force_model"), str
        ):
            route_key = router_overrides["force_model"]
            target_model_id = routed_map.get(route_key)
            if target_model_id:
                body.setdefault("metadata", {})
                body["metadata"]["router"] = {
                    "model": route_key,
                    "model_id": target_model_id,
                    "reason": "forced_model",
                    "classifier_model_id": None,
                    "elapsed_ms": 0,
                }
                self._log_message(f"Forced model '{route_key}' -> '{target_model_id}'")
                if target_model_id == body.get("model"):
                    self._log_message(
                        "Forced model equals incoming; returning unchanged"
                    )
                    return body
                body["model"] = target_model_id
                if self.valves.status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Mode: {route_key}",
                                "done": True,
                            },
                        }
                    )
                return body
            else:
                logger.warning(
                    "Forced model route key '%s' not found in routed_models; falling back.",
                    route_key,
                )
        # ------------------------------------------------

        # --- Analysis status ---
        if self.valves.status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "🤔 Chosing best mode...",
                        "done": False,
                    },
                }
            )
        # -----------------------

        # --- Non-vision classification ---
        valid_models = [k for k in routed_map.keys() if k != "vision"]
        if not valid_models:
            logger.warning(
                "No non-vision models configured in routed_models; skipping routing."
            )
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ Model selection failed",
                            "done": True,
                        },
                    }
                )
            return body

        result = await self._get_model_recommendation(
            body, valid_models, request=__request__, user=user
        )
        if result is None:
            self._log_message("Classifier unavailable; using fallback route")
        selected_model = result.get("model") if result else None

        # Map model -> model, with deterministic fallback
        target_model = None
        if selected_model in routed_map:
            target_model = routed_map[selected_model]
        else:
            target_model = routed_map.get(self.valves.fallback_route_key)
            if selected_model is not None:
                self._log_message(
                    None,
                    verbose_msg=f"Unknown model '{selected_model}' from classifier; falling back.",
                )

        if not target_model:
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ Model selection failed",
                            "done": True,
                        },
                    }
                )
            return body
        self._log_message(
            f"Selected model: '{target_model}'",
            verbose_msg=(
                f"model: {selected_model}, Reason: {result.get('reason')}, Elapsed: {result.get('elapsed_ms')}ms"
                if result
                else "fallback"
            ),
        )

        # Breadcrumb
        body.setdefault("metadata", {})
        body["metadata"]["router"] = {
            "model": selected_model if selected_model in routed_map else "fallback",
            "model_id": target_model,
            "reason": (result.get("reason") if result else "fallback"),
            "classifier_model_id": self.valves.router_model_id,
            "elapsed_ms": (result.get("elapsed_ms") if result else None),
        }

        # Same-model guard to avoid duplicate API call with mismatched metadata
        if target_model == body.get("model"):
            self._log_message(
                f"Selected model '{target_model}' equals incoming model; returning unchanged."
            )
            return body

        # Apply selection
        body["model"] = target_model

        # Optional reasoning message for UX (compact)
        if self.valves.show_reasoning and result and result.get("reason"):
            reasoning_message = (
                "<details>\n"
                "<summary>Model Selection</summary>\n"
                f"Type: {selected_model}\n"
                f"Model: {target_model}\n"
                f"Reason: {result.get('reason')}\n"
                "</details>"
            )
            await __event_emitter__(
                {"type": "message", "data": {"content": reasoning_message}}
            )

        if self.valves.status:
            route_key_for_ux = (
                selected_model
                if selected_model in routed_map
                else self.valves.fallback_route_key
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"💬 Mode selected: {route_key_for_ux}",
                        "done": True,
                    },
                }
            )
        self._log_message("Routing complete")

        return body
