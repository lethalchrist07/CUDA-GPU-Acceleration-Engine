import re
from typing import Tuple


def sanitize_prompt(prompt: str, max_chars: int = 8192) -> str:
    """
    Sanitize and enforce bounded resource limits on user prompts.
    Strips dangerous control codes while preserving markdown and code syntax.
    """
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")

    # Strip null bytes and non-printable control characters (except newline, carriage return, tab)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", prompt)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError("Prompt cannot be empty after sanitization")

    # Enforce safe length limit to prevent OOM
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]

    return cleaned


def validate_inference_parameters(temperature: float, max_tokens: int, context_length: int) -> Tuple[float, int, int]:
    """Validate and clamp inference hyperparameters within hardware-safe boundaries."""
    clamped_temp = max(0.0, min(float(temperature), 2.0))
    clamped_max_tokens = max(1, min(int(max_tokens), 2048))
    clamped_ctx = max(512, min(int(context_length), 8192))
    return clamped_temp, clamped_max_tokens, clamped_ctx


def safe_display_output(text: str) -> str:
    """
    Format output safely for terminal display without executing any embedded shell sequences.
    """
    if not isinstance(text, str):
        return ""
    # Strip terminal escape sequences (ANSI escape codes)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)
