from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from verifier.result import SandboxConfig


HARNESS_SOURCE = r'''
from __future__ import annotations

import builtins
import io
import json
import os
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import traceback


_ORIGINAL_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_OS_EXIT = os._exit
_SANDBOX_ROOT = os.getcwd()
_ALLOWED_PREFIXES = {
    os.path.abspath(_SANDBOX_ROOT),
    os.path.abspath(tempfile.gettempdir()),
    os.path.abspath(sys.prefix),
    os.path.abspath(sys.base_prefix),
}
for _path in sysconfig.get_paths().values():
    if isinstance(_path, str) and _path:
        _ALLOWED_PREFIXES.add(os.path.abspath(_path))


class _SandboxTimeout(BaseException):
    pass


def _json_default(value):
    return str(value)


def _blocked(*args, **kwargs):
    raise RuntimeError("operation disabled by verifier sandbox")


def _path_allowed(path) -> bool:
    raw_path = os.fspath(path)
    if raw_path in {"", os.devnull}:
        return True
    absolute = os.path.abspath(raw_path)
    if absolute in {"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr"}:
        return True
    for prefix in _ALLOWED_PREFIXES:
        if absolute == prefix or absolute.startswith(prefix + os.sep):
            return True
    return False


def _safe_open(file, mode="r", *args, **kwargs):
    if isinstance(file, int):
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)
    if _path_allowed(file):
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)
    raise PermissionError(f"filesystem restricted by verifier sandbox: {file}")


def _safe_os_open(path, flags, mode=0o777, *args, **kwargs):
    if _path_allowed(path):
        return _ORIGINAL_OS_OPEN(path, flags, mode, *args, **kwargs)
    raise PermissionError(f"filesystem restricted by verifier sandbox: {path}")


def _disable_network():
    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    socket.gethostbyname = _blocked


def _disable_process_creation():
    subprocess.Popen = _blocked
    os.system = _blocked
    for name in ("fork", "forkpty", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "posix_spawn"):
        if hasattr(os, name):
            setattr(os, name, _blocked)


def _restrict_filesystem():
    builtins.open = _safe_open
    io.open = _safe_open
    os.open = _safe_os_open


def _timeout_handler(signum, frame):
    raise _SandboxTimeout("wall-time limit exceeded")


def _decode_limited(file_obj, output_limit_bytes: int):
    file_obj.flush()
    size = file_obj.tell()
    file_obj.seek(0)
    data = file_obj.read(output_limit_bytes + 1)
    truncated = len(data) > output_limit_bytes or size > output_limit_bytes
    data = data[:output_limit_bytes]
    return data.decode("utf-8", errors="replace"), int(size), truncated


def _text_stream_for_fd(fd: int, mode: str):
    raw = os.fdopen(os.dup(fd), mode, closefd=True)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", line_buffering=True)


def _run_one(compiled, testcase: dict, payload: dict) -> dict:
    test_id = str(testcase.get("test_id") or "case")
    input_text = str(testcase.get("input") or "")
    wall_time_seconds = float(payload["wall_time_seconds"])
    output_limit_bytes = int(payload["output_limit_bytes"])
    original_threads = set(threading.enumerate())
    started_at = time.perf_counter()
    exit_code = 0
    timeout = False
    error_type = None
    error_message = None
    fatal = False

    with tempfile.TemporaryFile("w+b") as stdin_file, tempfile.TemporaryFile("w+b") as stdout_file, tempfile.TemporaryFile("w+b") as stderr_file:
        stdin_file.write(input_text.encode("utf-8", errors="replace"))
        stdin_file.flush()
        stdin_file.seek(0)
        os.dup2(stdin_file.fileno(), 0)
        os.dup2(stdout_file.fileno(), 1)
        os.dup2(stderr_file.fileno(), 2)
        sys.stdin = _text_stream_for_fd(0, "rb")
        sys.stdout = _text_stream_for_fd(1, "wb")
        sys.stderr = _text_stream_for_fd(2, "wb")
        sys.argv = ["solution.py"]
        globals_dict = {
            "__name__": "__main__",
            "__file__": "solution.py",
            "__builtins__": builtins.__dict__,
        }

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, wall_time_seconds)
        try:
            exec(compiled, globals_dict)
            deadline = started_at + wall_time_seconds
            for thread in set(threading.enumerate()) - original_threads:
                if thread is threading.current_thread():
                    continue
                remaining = max(0.0, deadline - time.perf_counter())
                if remaining <= 0:
                    raise _SandboxTimeout("thread did not finish before wall-time limit")
                thread.join(remaining)
                if thread.is_alive():
                    fatal = True
                    raise _SandboxTimeout("thread did not finish before wall-time limit")
        except _SandboxTimeout as exc:
            timeout = True
            error_type = type(exc).__name__
            error_message = str(exc)
            fatal = fatal or bool(set(threading.enumerate()) - original_threads)
        except SystemExit as exc:
            code = exc.code
            if code in (None, 0):
                exit_code = 0
            elif isinstance(code, int):
                exit_code = int(code)
                error_type = "SystemExit"
                error_message = str(code)
            else:
                exit_code = 1
                error_type = "SystemExit"
                error_message = str(code)
        except BaseException as exc:  # noqa: BLE001 - user code failures are classified as runtime errors.
            exit_code = 1
            error_type = type(exc).__name__
            error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except BaseException:
                    pass

        runtime_ms = int(round((time.perf_counter() - started_at) * 1000))
        stdout_text, stdout_size, stdout_truncated = _decode_limited(stdout_file, output_limit_bytes)
        stderr_text, stderr_size, stderr_truncated = _decode_limited(stderr_file, output_limit_bytes)

    return {
        "test_id": test_id,
        "exit_code": exit_code,
        "runtime_ms": runtime_ms,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_size": stdout_size,
        "stderr_size": stderr_size,
        "timeout": timeout,
        "output_truncated": stdout_truncated or stderr_truncated,
        "error_type": error_type,
        "error_message": error_message,
        "fatal": fatal,
    }


def main() -> None:
    payload_path = sys.argv[1]
    result_path = sys.argv[2]
    with _ORIGINAL_OPEN(payload_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("network_disabled", True):
        _disable_network()
    if payload.get("process_creation_disabled", True):
        _disable_process_creation()
    if payload.get("restricted_filesystem", True):
        _restrict_filesystem()

    results = []
    try:
        compiled = compile(str(payload["code"]), "solution.py", "exec")
    except BaseException as exc:  # noqa: BLE001
        with _ORIGINAL_OPEN(result_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "compile_error": {
                        "error_type": type(exc).__name__,
                        "error_message": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    },
                    "test_runs": [],
                },
                handle,
                default=_json_default,
            )
        return

    for testcase in payload["tests"]:
        result = _run_one(compiled, testcase, payload)
        results.append(result)
        if result["fatal"]:
            break

    with _ORIGINAL_OPEN(result_path, "w", encoding="utf-8") as handle:
        json.dump({"compile_error": None, "test_runs": results}, handle, default=_json_default)


if __name__ == "__main__":
    main()
'''


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _apply_resource_limits(config: SandboxConfig, test_count: int) -> None:
    try:
        import resource

        cpu_limit = max(1, int(math.ceil(float(config.cpu_time_seconds) * max(1, test_count))) + 1)
        memory_limit = max(64, int(config.memory_mb)) * 1024 * 1024
        file_limit = max(int(config.output_limit_bytes) * max(4, test_count), 16 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        if hasattr(resource, "RLIMIT_DATA"):
            resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, memory_limit))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (int(config.process_limit), int(config.process_limit)))
        if hasattr(resource, "RLIMIT_FSIZE"):
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    except Exception:
        pass
    try:
        os.setsid()
    except Exception:
        pass


def _payload(code: str, tests: list[dict[str, Any]], config: SandboxConfig) -> dict[str, Any]:
    return {
        "code": code,
        "tests": tests,
        "wall_time_seconds": float(config.wall_time_seconds),
        "output_limit_bytes": int(config.output_limit_bytes),
        "network_disabled": bool(config.network_disabled),
        "restricted_filesystem": bool(config.restricted_filesystem),
        "process_creation_disabled": True,
    }


def _python_bin(config: SandboxConfig) -> str:
    return config.python_executable or sys.executable


def _run_local(code: str, tests: list[dict[str, Any]], config: SandboxConfig) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="codereason-verifier-") as tmpdir:
        root = Path(tmpdir)
        harness_path = root / "harness.py"
        payload_path = root / "payload.json"
        result_path = root / "result.json"
        harness_path.write_text(HARNESS_SOURCE, encoding="utf-8")
        _write_json(payload_path, _payload(code, tests, config))

        whole_timeout = max(1.0, min(float(config.wall_time_seconds) * max(1, len(tests)) + 5.0, 60.0))
        try:
            completed = subprocess.run(
                [_python_bin(config), str(harness_path), str(payload_path), str(result_path)],
                cwd=root,
                env={
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONHASHSEED": "0",
                    "HOME": str(root),
                    "TMPDIR": str(root),
                    "PATH": os.environ.get("PATH", ""),
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=whole_timeout,
                preexec_fn=lambda: _apply_resource_limits(config, len(tests)) if os.name == "posix" else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "backend": "local",
                "protocol_error": "harness_timeout",
                "harness_exit_code": None,
                "harness_stdout_size": len(exc.stdout or b""),
                "harness_stderr_size": len(exc.stderr or b""),
                "test_runs": [],
            }

        if not result_path.exists():
            return {
                "backend": "local",
                "protocol_error": "missing_result",
                "harness_exit_code": completed.returncode,
                "harness_stdout_size": len(completed.stdout),
                "harness_stderr_size": len(completed.stderr),
                "test_runs": [],
            }

        result = _read_json(result_path)
        result.update(
            {
                "backend": "local",
                "protocol_error": None,
                "harness_exit_code": completed.returncode,
                "harness_stdout_size": len(completed.stdout),
                "harness_stderr_size": len(completed.stderr),
            }
        )
        return result


def _run_docker(code: str, tests: list[dict[str, Any]], config: SandboxConfig) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="codereason-verifier-") as tmpdir:
        root = Path(tmpdir)
        harness_path = root / "harness.py"
        payload_path = root / "payload.json"
        result_path = root / "result.json"
        harness_path.write_text(HARNESS_SOURCE, encoding="utf-8")
        _write_json(payload_path, _payload(code, tests, config))
        root.chmod(0o777)
        harness_path.chmod(0o644)
        payload_path.chmod(0o644)
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            f"{int(config.memory_mb)}m",
            "--pids-limit",
            str(int(config.process_limit)),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "-v",
            f"{root}:/sandbox:rw",
            "-w",
            "/sandbox",
            config.docker_image,
            "python",
            "/sandbox/harness.py",
            "/sandbox/payload.json",
            "/sandbox/result.json",
        ]
        whole_timeout = max(1.0, min(float(config.wall_time_seconds) * max(1, len(tests)) + 10.0, 90.0))
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=whole_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "backend": "docker",
                "protocol_error": "harness_timeout",
                "harness_exit_code": None,
                "harness_stdout_size": len(exc.stdout or b""),
                "harness_stderr_size": len(exc.stderr or b""),
                "test_runs": [],
            }
        if not result_path.exists():
            return {
                "backend": "docker",
                "protocol_error": "missing_result",
                "harness_exit_code": completed.returncode,
                "harness_stdout_size": len(completed.stdout),
                "harness_stderr_size": len(completed.stderr),
                "test_runs": [],
            }
        result = _read_json(result_path)
        result.update(
            {
                "backend": "docker",
                "protocol_error": None,
                "harness_exit_code": completed.returncode,
                "harness_stdout_size": len(completed.stdout),
                "harness_stderr_size": len(completed.stderr),
            }
        )
        return result


def run_python_in_sandbox(code: str, tests: list[dict[str, Any]], config: SandboxConfig) -> dict[str, Any]:
    if config.backend == "docker":
        return _run_docker(code, tests, config)
    if config.backend == "local":
        return _run_local(code, tests, config)
    raise ValueError(f"Unsupported sandbox backend: {config.backend}")


def normalization_notes(config: SandboxConfig) -> list[str]:
    backend_notes = [
        f"backend={config.backend}",
        f"wall_time_seconds_per_test={config.wall_time_seconds}",
        f"cpu_time_seconds_per_test={config.cpu_time_seconds}",
        f"memory_mb={config.memory_mb}",
        f"process_limit={config.process_limit}",
        f"output_limit_bytes_per_stream={config.output_limit_bytes}",
    ]
    if config.backend == "local":
        backend_notes.append("network_disabled=python_stdlib_socket_block")
        backend_notes.append("restricted_filesystem=best_effort_temp_cwd_with_open_guards")
    else:
        backend_notes.append("network_disabled=docker_network_none")
        backend_notes.append("restricted_filesystem=docker_read_only_root_plus_tmpfs")
    return backend_notes
