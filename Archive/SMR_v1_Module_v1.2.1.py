"""
title: Smart Model Router (SMR)
description: A minimal router for Open WebUI that selects a target model per request.
author: Cody
version: 1.2.1
date: 2025-08-17
dev_note: "Cody: Major refactor to use a required, admin-provided prompt as the single source of truth. Added valves for history, timeout, and temperature."
"""

import logging
import json
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
logger = logging.getLogger("ams_smr")
logger.propagate = False
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class Filter:
    """
    Minimal Smart Model Router:
    - Vision short-circuit: route any image-containing request directly to routed_models["vision"].
    - Non-vision: classify to a model using a local Qwen model (router_model_id), then map model -> model via routed_models.
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
            default="qwen2.5:14b",
            description="Model ID used ONLY to classify (routing decisions). Should be a fast.",
        )
        routed_models: str = Field(
            default=(
                "{\n"
                '  "vision": "gpt-4.1-mini",\n'
                '  "simple": "qwen2.5:14b",\n'
                '  "nuanced": "gpt-5-chat-latest",\n'
                '  "advanced": "gpt-5"\n'
                "}"
            ),
            description="JSON object mapping models -> model IDs. Example:",
        )
        router_system_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Classifier system prompt template. Must be provided by admin. Should contain {models} and {user_request} placeholders.",
        )
        history_window: int = Field(
            default=5,
            description="Number of recent messages to include in the classifier's context.",
        )
        classifier_timeout_seconds: int = Field(
            default=3,
            description="Timeout in seconds for the classifier model's response.",
        )
        classifier_temperature: float = Field(
            default=0.0,
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

    def _format_log_content(self, text: str, max_len: int = 200) -> str:
        """
        Make text single-line and bounded for clean logs.
        """
        if not text:
            return ""
        single_line = str(text).replace("\n", " ").strip()
        return (
            single_line
            if len(single_line) <= max_len
            else single_line[: max_len - 3] + "..."
        )

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
            return any(
                (isinstance(item, dict) and item.get("type") == "image_url")
                for item in content
            )
        # Back-compat: some sources may use 'images' key
        return bool(last_user_message_dict.get("images"))

    def _load_routed_models(self) -> Dict[str, str]:
        """Load and validate routed_models mapping from valves."""
        try:
            data = json.loads(self.valves.routed_models or "{}")
            if not isinstance(data, dict):
                logger.warning("routed_models is not a JSON object; got %s", type(data))
                return {}
            # Keep only string -> string pairs
            cleaned = {
                str(k): str(v)
                for k, v in data.items()
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
            }
            if not cleaned:
                logger.warning("routed_models mapping is empty after validation")
            return cleaned
        except Exception as e:
            logger.error("Failed to parse routed_models JSON: %s", str(e))
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

        messages = body.get("messages", [])[-self.valves.history_window :]
        transcript_lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = self._flatten_text(msg.get("content"))
            if text:
                transcript_lines.append(f"- {role}: {text}")
        transcript = "\n".join(transcript_lines)
        last_user = get_last_user_message(body.get("messages", []) or []) or ""

        models_json = json.dumps(models, ensure_ascii=False)
        user_request_content = (
            "Recent conversation (text-only, most recent last):\n"
            f"{transcript}\n\n"
            "Final user request (focus primarily on this):\n"
            f"{last_user}\n"
        )

        system_prompt = self.valves.router_system_prompt.format(
            models=models_json, user_request=user_request_content
        )

        payload = {
            "model": self.valves.router_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Return JSON only."},
            ],
            "stream": False,
            "temperature": self.valves.classifier_temperature,
            "metadata": {"user_id": user.id if user else None},
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
            data = json.loads(raw)

            model = data.get("model")
            confidence = None
            reason = data.get("reason")

            if not isinstance(model, str) or model not in models:
                self._log_message(
                    None,
                    verbose_msg=f"Classifier returned invalid or unknown model: {model!r}",
                )
                return {
                    "model": None,
                    "confidence": None,
                    "reason": reason,
                    "elapsed_ms": elapsed_ms,
                }

            # Type hygiene
            confidence = None
            if reason is not None and not isinstance(reason, str):
                reason = None

            self._log_message(
                f"Classifier result: model='{model}', elapsed_ms={elapsed_ms}",
                verbose_msg=f"reason={reason}",
            )

            return {
                "model": model,
                "confidence": None,
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

        # Remove unsupported/legacy options at top level
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

        # Remove options nested under metadata if present
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
                            "description": "Routing image request to vision model",
                            "done": True,
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
                "confidence": None,
                "classifier_model_id": None,
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
                    "confidence": None,
                    "classifier_model_id": None,
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
                        "description": "Analyzing request to select best model...",
                        "done": False,
                    },
                }
            )
        # -----------------------

        # --- Non-vision classification via Qwen ---
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
            target_model = routed_map.get("nuanced")
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
            "confidence": None,
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
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Mode: {selected_model}",
                        "done": True,
                    },
                }
            )
        self._log_message("Routing complete")

        return body
