from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class VerifierStatus(StrEnum):
    AC = "AC"
    WA = "WA"
    CE = "CE"
    RE = "RE"
    TLE = "TLE"


@dataclass(frozen=True)
class ExtractedCode:
    code: str
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntaxCheckResult:
    ok: bool
    method: str
    error_type: str | None = None
    error_message: str | None = None
    line_no: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxConfig:
    backend: str = "local"
    python_executable: str | None = None
    docker_image: str = "python:3.11-slim"
    wall_time_seconds: float = 2.0
    cpu_time_seconds: int = 2
    memory_mb: int = 1024
    process_limit: int = 64
    output_limit_bytes: int = 1_000_000
    network_disabled: bool = True
    restricted_filesystem: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestRunResult:
    test_id: str
    status: VerifierStatus
    passed: bool
    exit_code: int | None
    runtime_ms: int
    stdout_size: int
    stderr_size: int
    timeout: bool = False
    output_truncated: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["status"] = self.status.value
        return output


@dataclass(frozen=True)
class VerificationResult:
    status: VerifierStatus
    compile_success: bool
    runtime_success: bool
    timeout: bool
    passed: int
    total: int
    pass_rate: float
    exit_code: int | None
    runtime_ms: int
    stdout_size: int
    stderr_size: int
    extraction_strategy: str
    sandbox_backend: str
    normalization_policy: list[str]
    sandbox: dict[str, Any]
    syntax_error: dict[str, Any] | None = None
    test_results: list[TestRunResult] = field(default_factory=list)

    def to_dict(self, *, include_tests: bool = True) -> dict[str, Any]:
        output = asdict(self)
        output["status"] = self.status.value
        if include_tests:
            output["test_results"] = [item.to_dict() for item in self.test_results]
        else:
            output.pop("test_results", None)
        return output
