from __future__ import annotations

import ast
import py_compile
import tempfile
from pathlib import Path
from typing import Any

from verifier.extract_code import extract_python_code
from verifier.judge import NORMALIZATION_POLICY, judge_stdout
from verifier.result import SandboxConfig, SyntaxCheckResult, TestRunResult, VerificationResult, VerifierStatus
from verifier.sandbox import normalization_notes, run_python_in_sandbox


def _syntax_check(code: str) -> SyntaxCheckResult:
    try:
        ast.parse(code, filename="solution.py")
    except SyntaxError as exc:
        return SyntaxCheckResult(
            ok=False,
            method="ast.parse",
            error_type=type(exc).__name__,
            error_message=exc.msg,
            line_no=exc.lineno,
        )

    with tempfile.TemporaryDirectory(prefix="codereason-syntax-") as tmpdir:
        source_path = Path(tmpdir) / "solution.py"
        source_path.write_text(code, encoding="utf-8")
        try:
            py_compile.compile(str(source_path), doraise=True)
        except py_compile.PyCompileError as exc:
            return SyntaxCheckResult(
                ok=False,
                method="py_compile",
                error_type=type(exc).__name__,
                error_message=str(exc),
                line_no=None,
            )
    return SyntaxCheckResult(ok=True, method="ast.parse+py_compile")


def _coerce_tests(tests: list[dict[str, Any]]) -> list[dict[str, str]]:
    coerced: list[dict[str, str]] = []
    for index, testcase in enumerate(tests):
        test_id = testcase.get("test_id")
        coerced.append(
            {
                "test_id": str(test_id) if test_id is not None else f"case_{index:04d}",
                "input": str(testcase.get("input") or ""),
                "output": str(testcase.get("output") or ""),
            }
        )
    return coerced


def _result_for_ce(
    *,
    total: int,
    extraction_strategy: str,
    sandbox_config: SandboxConfig,
    syntax: SyntaxCheckResult,
) -> VerificationResult:
    return VerificationResult(
        status=VerifierStatus.CE,
        compile_success=False,
        runtime_success=False,
        timeout=False,
        passed=0,
        total=total,
        pass_rate=0.0,
        exit_code=None,
        runtime_ms=0,
        stdout_size=0,
        stderr_size=0,
        extraction_strategy=extraction_strategy,
        sandbox_backend=sandbox_config.backend,
        normalization_policy=NORMALIZATION_POLICY,
        sandbox=sandbox_config.to_dict(),
        syntax_error=syntax.to_dict(),
        test_results=[],
    )


def _result_for_protocol_error(
    *,
    total: int,
    extraction_strategy: str,
    sandbox_config: SandboxConfig,
    sandbox_result: dict[str, Any],
) -> VerificationResult:
    timeout = sandbox_result.get("protocol_error") == "harness_timeout"
    status = VerifierStatus.TLE if timeout else VerifierStatus.RE
    failure = TestRunResult(
        test_id="__harness__",
        status=status,
        passed=False,
        exit_code=sandbox_result.get("harness_exit_code"),
        runtime_ms=0,
        stdout_size=int(sandbox_result.get("harness_stdout_size", 0)),
        stderr_size=int(sandbox_result.get("harness_stderr_size", 0)),
        timeout=timeout,
        error_type=str(sandbox_result.get("protocol_error") or "protocol_error"),
        error_message="sandbox harness did not produce a complete result",
    )
    return VerificationResult(
        status=status,
        compile_success=True,
        runtime_success=False,
        timeout=timeout,
        passed=0,
        total=total,
        pass_rate=0.0,
        exit_code=sandbox_result.get("harness_exit_code"),
        runtime_ms=0,
        stdout_size=int(sandbox_result.get("harness_stdout_size", 0)),
        stderr_size=int(sandbox_result.get("harness_stderr_size", 0)),
        extraction_strategy=extraction_strategy,
        sandbox_backend=sandbox_config.backend,
        normalization_policy=NORMALIZATION_POLICY,
        sandbox={**sandbox_config.to_dict(), "backend_notes": normalization_notes(sandbox_config)},
        syntax_error=None,
        test_results=[failure],
    )


def _status_from_runs(test_results: list[TestRunResult], passed: int, total: int) -> VerifierStatus:
    if any(item.timeout for item in test_results):
        return VerifierStatus.TLE
    if any(item.status == VerifierStatus.RE for item in test_results):
        return VerifierStatus.RE
    if passed != total:
        return VerifierStatus.WA
    return VerifierStatus.AC


def _run_to_test_result(run: dict[str, Any], expected_output: str) -> TestRunResult:
    test_id = str(run.get("test_id") or "case")
    timeout = bool(run.get("timeout"))
    output_truncated = bool(run.get("output_truncated"))
    exit_code = run.get("exit_code")
    error_type = run.get("error_type")
    error_message = run.get("error_message")
    actual_output = str(run.get("stdout") or "")
    if timeout:
        status = VerifierStatus.TLE
        passed = False
    elif output_truncated:
        status = VerifierStatus.RE
        passed = False
        error_type = error_type or "OutputLimitExceeded"
        error_message = error_message or "stdout/stderr exceeded output limit"
    elif exit_code not in (0, None) or error_type:
        status = VerifierStatus.RE
        passed = False
    elif judge_stdout(actual_output, expected_output):
        status = VerifierStatus.AC
        passed = True
    else:
        status = VerifierStatus.WA
        passed = False

    return TestRunResult(
        test_id=test_id,
        status=status,
        passed=passed,
        exit_code=exit_code,
        runtime_ms=int(run.get("runtime_ms") or 0),
        stdout_size=int(run.get("stdout_size") or 0),
        stderr_size=int(run.get("stderr_size") or 0),
        timeout=timeout,
        output_truncated=output_truncated,
        error_type=error_type,
        error_message=error_message,
    )


def verify_code(
    code: str,
    tests: list[dict[str, Any]],
    *,
    sandbox_config: SandboxConfig | None = None,
    extraction_strategy: str = "direct",
) -> VerificationResult:
    sandbox_config = sandbox_config or SandboxConfig()
    coerced_tests = _coerce_tests(tests)
    syntax = _syntax_check(code)
    if not syntax.ok:
        return _result_for_ce(
            total=len(coerced_tests),
            extraction_strategy=extraction_strategy,
            sandbox_config=sandbox_config,
            syntax=syntax,
        )

    sandbox_result = run_python_in_sandbox(code, coerced_tests, sandbox_config)
    if sandbox_result.get("compile_error"):
        syntax_error = SyntaxCheckResult(
            ok=False,
            method="sandbox_compile",
            error_type=sandbox_result["compile_error"].get("error_type"),
            error_message=sandbox_result["compile_error"].get("error_message"),
        )
        return _result_for_ce(
            total=len(coerced_tests),
            extraction_strategy=extraction_strategy,
            sandbox_config=sandbox_config,
            syntax=syntax_error,
        )
    if sandbox_result.get("protocol_error"):
        return _result_for_protocol_error(
            total=len(coerced_tests),
            extraction_strategy=extraction_strategy,
            sandbox_config=sandbox_config,
            sandbox_result=sandbox_result,
        )

    test_results: list[TestRunResult] = []
    runs = sandbox_result.get("test_runs", [])
    for index, run in enumerate(runs):
        expected = coerced_tests[index]["output"] if index < len(coerced_tests) else ""
        test_results.append(_run_to_test_result(run, expected))
    if len(test_results) < len(coerced_tests):
        seen_ids = {item.test_id for item in test_results}
        for testcase in coerced_tests:
            if testcase["test_id"] in seen_ids:
                continue
            test_results.append(
                TestRunResult(
                    test_id=testcase["test_id"],
                    status=VerifierStatus.TLE,
                    passed=False,
                    exit_code=None,
                    runtime_ms=0,
                    stdout_size=0,
                    stderr_size=0,
                    timeout=True,
                    error_type="HarnessStoppedEarly",
                    error_message="sandbox stopped before running this testcase",
                )
            )

    passed = sum(1 for item in test_results if item.passed)
    total = len(coerced_tests)
    status = _status_from_runs(test_results, passed, total)
    timeout = any(item.timeout for item in test_results)
    runtime_success = not timeout and not any(item.status == VerifierStatus.RE for item in test_results)
    return VerificationResult(
        status=status,
        compile_success=True,
        runtime_success=runtime_success,
        timeout=timeout,
        passed=passed,
        total=total,
        pass_rate=(passed / total if total else 0.0),
        exit_code=next((item.exit_code for item in test_results if item.status != VerifierStatus.AC), 0),
        runtime_ms=sum(item.runtime_ms for item in test_results),
        stdout_size=sum(item.stdout_size for item in test_results),
        stderr_size=sum(item.stderr_size for item in test_results),
        extraction_strategy=extraction_strategy,
        sandbox_backend=str(sandbox_result.get("backend") or sandbox_config.backend),
        normalization_policy=NORMALIZATION_POLICY,
        sandbox={**sandbox_config.to_dict(), "backend_notes": normalization_notes(sandbox_config)},
        syntax_error=None,
        test_results=test_results,
    )


def verify(
    response: str,
    tests: list[dict[str, Any]],
    *,
    sandbox_config: SandboxConfig | None = None,
) -> VerificationResult:
    extracted = extract_python_code(response)
    return verify_code(
        extracted.code,
        tests,
        sandbox_config=sandbox_config,
        extraction_strategy=extracted.strategy,
    )
