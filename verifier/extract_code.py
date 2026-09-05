from __future__ import annotations

import re

from verifier.result import ExtractedCode


FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n(?P<code>.*?)(?:\n```|```)", re.DOTALL)


def _strip_response(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("<think>") and "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].strip()
    return stripped


def extract_python_code(response: str) -> ExtractedCode:
    stripped = _strip_response(response)
    if not stripped:
        return ExtractedCode(code="", strategy="empty")

    matches = list(FENCE_RE.finditer(stripped))
    for match in matches:
        language = match.group("lang").strip().lower()
        if language in {"python", "py", "python3"}:
            return ExtractedCode(code=match.group("code").strip() + "\n", strategy=f"fenced:{language}")
    if matches:
        return ExtractedCode(code=matches[0].group("code").strip() + "\n", strategy="fenced:untyped")

    if stripped.startswith("```") and stripped.endswith("```"):
        body = stripped.strip("`").strip()
        return ExtractedCode(code=body + "\n", strategy="fence-stripped")

    return ExtractedCode(code=stripped + "\n", strategy="direct")
