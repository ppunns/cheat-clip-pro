import logging
import json
import re
from typing import List, Dict, Any, Optional
import httpx
from backend.schemas.analyze import VideoAnalysis

logger = logging.getLogger("cheatclip.openrouter")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Curated list of popular, high-capability text LLMs for transcript analysis
POPULAR_OPENROUTER_MODELS = [
    {
        "id": "google/gemini-2.5-flash",
        "name": "Google: Gemini 2.5 Flash",
        "context_length": 1000000,
        "description": "Recommended - ultra fast, 1M context window, ideal for long transcripts",
        "family": "google",
        "recommended": True
    },
    {
        "id": "google/gemini-2.0-flash-001",
        "name": "Google: Gemini 2.0 Flash",
        "context_length": 1000000,
        "description": "High-speed multimodal LLM with 1M context",
        "family": "google",
        "recommended": True
    },
    {
        "id": "anthropic/claude-3.5-sonnet",
        "name": "Anthropic: Claude 3.5 Sonnet",
        "context_length": 200000,
        "description": "State-of-the-art reasoning, viral hook generation & nuanced comprehension",
        "family": "anthropic",
        "recommended": True
    },
    {
        "id": "anthropic/claude-3.5-haiku",
        "name": "Anthropic: Claude 3.5 Haiku",
        "context_length": 200000,
        "description": "Blazing fast inference, cost-effective transcript analysis",
        "family": "anthropic"
    },
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek: DeepSeek V3",
        "context_length": 64000,
        "description": "Exceptional value and high quality viral scoring",
        "family": "deepseek",
        "recommended": True
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Meta: Llama 3.3 70B Instruct",
        "context_length": 128000,
        "description": "Top-tier open weights model for instruction following",
        "family": "meta-llama"
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "OpenAI: GPT-4o Mini",
        "context_length": 128000,
        "description": "Fast and intelligent lightweight flagship",
        "family": "openai"
    },
    {
        "id": "openai/gpt-4o",
        "name": "OpenAI: GPT-4o",
        "context_length": 128000,
        "description": "OpenAI flagship intelligence",
        "family": "openai"
    },
    {
        "id": "qwen/qwen-2.5-72b-instruct",
        "name": "Qwen: Qwen 2.5 72B Instruct",
        "context_length": 128000,
        "description": "Outstanding multilingual & Indonesian transcript comprehension",
        "family": "qwen"
    },
    {
        "id": "mistralai/mistral-large-2411",
        "name": "Mistral: Mistral Large 2",
        "context_length": 128000,
        "description": "Enterprise-grade reasoning and reasoning quality",
        "family": "mistralai"
    },
    {
        "id": "google/gemini-2.5-pro",
        "name": "Google: Gemini 2.5 Pro",
        "context_length": 1000000,
        "description": "Maximum reasoning depth for complex podcasts and long interviews",
        "family": "google"
    }
]


def get_openrouter_fallback_chain(requested_model: str) -> List[str]:
    """
    Builds a fallback chain within the SAME model family (e.g., Claude -> Claude, Gemini -> Gemini, Llama -> Llama)
    to prevent cross-family mismatches per user guidelines.
    """
    requested_model = (requested_model or "google/gemini-2.5-flash").strip()
    chain = [requested_model]
    req_lower = requested_model.lower()

    if "google" in req_lower or "gemini" in req_lower:
        family_candidates = [
            "google/gemini-2.5-flash",
            "google/gemini-2.0-flash-001",
            "google/gemini-flash-1.5",
            "google/gemini-2.5-pro"
        ]
    elif "anthropic" in req_lower or "claude" in req_lower:
        family_candidates = [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-opus"
        ]
    elif "meta-llama" in req_lower or "llama" in req_lower:
        family_candidates = [
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct"
        ]
    elif "openai" in req_lower or "gpt" in req_lower:
        family_candidates = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-4-turbo"
        ]
    elif "deepseek" in req_lower:
        family_candidates = [
            "deepseek/deepseek-chat",
            "deepseek/deepseek-coder"
        ]
    elif "qwen" in req_lower:
        family_candidates = [
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2.5-32b-instruct",
            "qwen/qwen-2.5-7b-instruct"
        ]
    elif "mistral" in req_lower:
        family_candidates = [
            "mistralai/mistral-large-2411",
            "mistralai/mistral-small-2409",
            "mistralai/mistral-7b-instruct"
        ]
    else:
        family_candidates = []

    for cand in family_candidates:
        if cand not in chain:
            chain.append(cand)

    return chain[:3]


async def fetch_openrouter_available_models(api_key: str = "") -> List[Dict[str, Any]]:
    """
    Fetches models from OpenRouter, filtering exclusively for text-generation LLMs
    with adequate context window (>=16k) suitable for long video transcript analysis.
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(OPENROUTER_MODELS_URL, headers=headers)
            if resp.status_code == 200:
                raw_models = resp.json().get("data", [])
                
                # Exclude image/audio/video generation models
                excluded_keywords = [
                    "image", "flux", "diffusion", "dall-e", "midjourney", 
                    "audio", "tts", "whisper", "embedding", "rerank", "moderation", "vision-preview"
                ]

                filtered_map: Dict[str, Dict[str, Any]] = {}
                for m in raw_models:
                    m_id = m.get("id", "")
                    m_name = m.get("name", m_id)
                    context_len = m.get("context_length") or 0
                    
                    if context_len < 16000:
                        continue
                    
                    m_id_lower = m_id.lower()
                    if any(ex in m_id_lower for ex in excluded_keywords):
                        continue
                    
                    family = m_id.split("/")[0] if "/" in m_id else "other"
                    filtered_map[m_id] = {
                        "id": m_id,
                        "name": m_name,
                        "context_length": context_len,
                        "description": m.get("description", "")[:120],
                        "family": family,
                        "recommended": False
                    }

                # Prioritize curated list at the top
                result_list: List[Dict[str, Any]] = []
                for pop in POPULAR_OPENROUTER_MODELS:
                    if pop["id"] in filtered_map:
                        merged = {**filtered_map[pop["id"]], **pop}
                        result_list.append(merged)
                        del filtered_map[pop["id"]]
                    else:
                        result_list.append(pop)

                # Append other high context models
                for remaining_model in list(filtered_map.values())[:30]:
                    result_list.append(remaining_model)

                return result_list
    except Exception as exc:
        logger.warning(f"Error fetching live OpenRouter models: {exc}. Returning curated defaults.")

    return POPULAR_OPENROUTER_MODELS


def resolve_chat_completions_url(api_endpoint: Optional[str] = None) -> str:
    """Resolves a base OpenAI-compatible endpoint (e.g. https://.../v1) to its /chat/completions URL."""
    if not api_endpoint or not api_endpoint.strip():
        return OPENROUTER_API_URL
    ep = api_endpoint.strip().rstrip("/")
    if ep.endswith("/chat/completions"):
        return ep
    if ep.endswith("/v1"):
        return f"{ep}/chat/completions"
    return f"{ep}/v1/chat/completions"


def resolve_models_url(api_endpoint: Optional[str] = None) -> str:
    """Resolves a base OpenAI-compatible endpoint (e.g. https://.../v1) to its /models URL."""
    if not api_endpoint or not api_endpoint.strip():
        return OPENROUTER_MODELS_URL
    ep = api_endpoint.strip().rstrip("/")
    if ep.endswith("/chat/completions"):
        ep = ep[:-len("/chat/completions")]
    if ep.endswith("/models"):
        return ep
    if ep.endswith("/v1"):
        return f"{ep}/models"
    return f"{ep}/v1/models"


def _extract_content_from_raw_response(raw_text: str) -> str:
    """
    Extracts the assistant message content from an HTTP 200 response body.
    Supports:
    1. Standard OpenAI JSON: {"choices": [{"message": {"content": "..."}}]}
    2. SSE stream body: multiple 'data: {...}' lines with delta.content or message.content
    3. Alternative JSON shapes: Anthropic content blocks, {"content": "..."}, {"response": "..."}, or direct VideoAnalysis JSON
    4. Raw plain-text / markdown JSON output directly from custom endpoints
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    # 1. Try parsing as a single JSON object
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # If the endpoint returned the VideoAnalysis JSON directly
            if "clips" in data and isinstance(data["clips"], list):
                return text

            choices = data.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0] or {}
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta") or {}
                    if isinstance(msg, dict) and msg.get("content"):
                        c = msg.get("content")
                        if isinstance(c, str):
                            return c.strip()
                        if isinstance(c, list):
                            return "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in c
                            ).strip()
                    if isinstance(first.get("text"), str) and first["text"].strip():
                        return first["text"].strip()

            # Anthropic / generic content formats
            for key in ("content", "output_text", "response", "text", "result"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, list):
                    joined = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in val
                    ).strip()
                    if joined:
                        return joined
    except Exception:
        pass

    # 2. Check if response body is Server-Sent Events (SSE: "data: {...}")
    if "data:" in text:
        chunks: List[str] = []
        for line in text.splitlines():
            line_s = line.strip()
            if not line_s.startswith("data:"):
                continue
            payload_str = line_s[5:].strip()
            if not payload_str or payload_str == "[DONE]":
                continue
            try:
                evt = json.loads(payload_str)
                if not isinstance(evt, dict):
                    continue
                choices = evt.get("choices")
                if isinstance(choices, list) and choices:
                    c0 = choices[0] or {}
                    delta = c0.get("delta") or c0.get("message") or {}
                    if isinstance(delta, dict):
                        piece = delta.get("content")
                        if isinstance(piece, str):
                            chunks.append(piece)
                    elif isinstance(c0.get("text"), str):
                        chunks.append(c0["text"])
                elif isinstance(evt.get("content"), str):
                    chunks.append(evt["content"])
                elif isinstance(evt.get("delta"), dict) and isinstance(evt["delta"].get("text"), str):
                    chunks.append(evt["delta"]["text"])
            except Exception:
                continue
        if chunks:
            return "".join(chunks).strip()

    # 3. If the raw body itself contains JSON with "clips", return it directly
    if '"clips"' in text and ("{" in text and "}" in text):
        return text

    return ""


def _parse_json_leniently(content: str) -> Optional[Dict[str, Any]]:
    """Extracts and repairs JSON from LLM text output (handles code fences, trailing commas, or truncation)."""
    content_clean = re.sub(r'<think>[\s\S]*?</think>', '', content or '').strip()
    if not content_clean:
        content_clean = (content or "").strip()
    if not content_clean:
        return None

    candidates = [content_clean]
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content_clean)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    bracket_match = re.search(r'\{[\s\S]*\}', content_clean)
    if bracket_match:
        candidates.append(bracket_match.group(0).strip())

    for cand in candidates:
        for variant in (cand, re.sub(r',\s*([\]}])', r'\1', cand)):
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict) and "clips" in parsed:
                    return parsed
            except Exception:
                pass

    # Truncated JSON recovery
    for cand in candidates:
        try:
            last_brace = cand.rfind("}")
            if last_brace != -1 and '"clips"' in cand:
                trunc = cand[:last_brace + 1]
                if not trunc.endswith("]}"):
                    trunc += "}" if trunc.endswith("]") else "]}"
                trunc = re.sub(r',\s*([\]}])', r'\1', trunc)
                parsed = json.loads(trunc)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass

    # Regex clip block recovery
    try:
        clips_list = []
        summary_m = re.search(r'"summary"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', content_clean)
        summary = summary_m.group(1) if summary_m else ""
        raw_blocks = re.findall(r'\{[^{}]*?"(?:title|start_time)"[^{}]*?\}', content_clean, re.DOTALL)
        for block in raw_blocks:
            try:
                c_obj = json.loads(re.sub(r',\s*([\]}])', r'\1', block))
                if isinstance(c_obj, dict):
                    clips_list.append(c_obj)
            except Exception:
                title_m = re.search(r'"title"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', block)
                start_m = re.search(r'"start_time"\s*:\s*([\d\.]+)', block)
                end_m = re.search(r'"end_time"\s*:\s*([\d\.]+)', block)
                score_m = re.search(r'"virality_score"\s*:\s*(\d+)', block)
                if start_m and end_m:
                    clips_list.append({
                        "title": title_m.group(1) if title_m else "Viral Highlight",
                        "start_time": float(start_m.group(1)),
                        "end_time": float(end_m.group(1)),
                        "hook_time": float(start_m.group(1)),
                        "virality_score": int(score_m.group(1)) if score_m else 85,
                        "key_quotes": [],
                    })
        if clips_list:
            return {"summary": summary, "clips": clips_list}
    except Exception:
        pass

    return None


async def analyze_with_openrouter(
    api_key: str,
    model_name: str,
    prompt: str,
    timeout: float = 120.0,
    api_endpoint: Optional[str] = None,
    system_instruction_override: Optional[str] = None
) -> VideoAnalysis:
    """
    Executes a structured JSON video analysis request against an OpenAI-compatible endpoint or OpenRouter.
    """
    target_url = resolve_chat_completions_url(api_endpoint)
    is_custom = bool(api_endpoint and api_endpoint.strip())
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "HTTP-Referer": "https://cheatclip.pro",
        "X-Title": "Cheat Clip Pro",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    system_instruction = (
        "You are an expert viral video clipper and AI retention analyst. "
        "Analyze the provided video transcript and select the highest engagement clips. "
        "Strict rules:\n"
        "1. STRICT ZERO-TRANSLATION RULE: Output titles, key_quotes, and summary in the EXACT SAME LANGUAGE as the video transcript.\n"
        "2. ZERO FIRST-PERSON PRONOUNS: Never use 'I', 'me', 'my', 'aku', 'saya' in titles or suggestions. Attribute the quote or insight to the speaker/host by name or use third-person objective framing.\n"
        "3. You MUST respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "summary": "1-2 sentence video summary in the exact transcript language, followed by 2-4 hashtags",\n'
        '  "clips": [\n'
        "    {\n"
        '      "title": "Snappy title under 8 words attributing speaker",\n'
        '      "start_time": 10.0,\n'
        '      "end_time": 45.0,\n'
        '      "hook_time": 12.0,\n'
        '      "virality_score": 92,\n'
        '      "key_quotes": ["quote 1", "quote 2"],\n'
        '      "title_suggestion": "Alternative viral hook title",\n'
        '      "caption_suggestion": "Social media post caption",\n'
        '      "hashtag_suggestion": "#hashtag1 #hashtag2"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    if system_instruction_override:
        system_instruction = system_instruction_override

    base_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    # Build payload variants: custom endpoints often work best without response_format or with stream=True
    payload_attempts = []
    if is_custom:
        payload_attempts.append({
            "model": model_name,
            "messages": base_messages,
            "temperature": 0.2,
            "stream": False,
        })
        payload_attempts.append({
            "model": model_name,
            "messages": base_messages,
            "temperature": 0.2,
            "stream": True,
        })
    else:
        payload_attempts.append({
            "model": model_name,
            "messages": base_messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        })
        payload_attempts.append({
            "model": model_name,
            "messages": base_messages,
            "temperature": 0.2,
        })

    last_status = 0
    last_raw_text = ""

    http_timeout = httpx.Timeout(timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=http_timeout, verify=False, follow_redirects=True) as client:
        for attempt_idx, pld in enumerate(payload_attempts):
            resp = await client.post(target_url, headers=headers, json=pld)
            last_status = resp.status_code
            last_raw_text = resp.text or ""

            if resp.status_code != 200:
                logger.warning(
                    f"AI endpoint {target_url} attempt {attempt_idx + 1} returned HTTP {resp.status_code}: {last_raw_text[:250]!r}"
                )
                continue

            content = _extract_content_from_raw_response(last_raw_text)
            if not content:
                logger.warning(
                    f"AI endpoint {target_url} attempt {attempt_idx + 1} returned HTTP 200 but empty content. Raw preview: {last_raw_text[:250]!r}"
                )
                continue

            parsed_dict = _parse_json_leniently(content)
            if isinstance(parsed_dict, dict):
                return VideoAnalysis.model_validate(parsed_dict)
            else:
                logger.warning(
                    f"AI endpoint {target_url} attempt {attempt_idx + 1} content could not be parsed as JSON: {content[:250]!r}"
                )

        # If all attempts failed, raise a clear diagnostic message
        if last_status != 200:
            err_msg = last_raw_text[:300]
            try:
                err_json = json.loads(last_raw_text)
                err_obj = err_json.get("error", {})
                err_msg = err_obj.get("message", err_msg) if isinstance(err_obj, dict) else str(err_obj or err_msg)
            except Exception:
                pass
            raise RuntimeError(f"AI Endpoint Error (HTTP {last_status}): {err_msg}")

        preview = (last_raw_text or "").strip()[:250]
        raise ValueError(
            f"Endpoint {target_url} (model '{model_name}') returned an empty or non-JSON response: {preview or '<empty body>'}"
        )
