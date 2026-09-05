from __future__ import annotations


NORMALIZATION_POLICY = [
    "Decode stdout/stderr as UTF-8 with replacement for invalid bytes.",
    "Convert CRLF and CR line endings to LF.",
    "Strip trailing spaces and tabs on each line.",
    "Remove trailing empty lines at end of output.",
    "Compare the remaining lines exactly; internal whitespace and line order are preserved.",
]


def normalize_output(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def judge_stdout(actual: str, expected: str) -> bool:
    return normalize_output(actual) == normalize_output(expected)
