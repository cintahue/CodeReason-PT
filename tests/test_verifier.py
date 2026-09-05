from __future__ import annotations

import unittest

from verifier.executor import verify
from verifier.result import SandboxConfig, VerifierStatus


TESTS = [{"test_id": "case_0", "input": "2 3\n", "output": "5\n"}]


class VerifierRegressionTest(unittest.TestCase):
    def sandbox_config(self) -> SandboxConfig:
        return SandboxConfig(wall_time_seconds=0.3, cpu_time_seconds=1, memory_mb=256, output_limit_bytes=100_000)

    def test_correct_program_is_ac(self) -> None:
        result = verify(
            "import sys\nnums=list(map(int, sys.stdin.read().split()))\nprint(sum(nums))\n",
            TESTS,
            sandbox_config=self.sandbox_config(),
        )
        self.assertEqual(result.status, VerifierStatus.AC)
        self.assertTrue(result.compile_success)
        self.assertTrue(result.runtime_success)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.total, 1)

    def test_syntax_error_is_ce(self) -> None:
        result = verify("def broken(:\n    pass\n", TESTS, sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.CE)
        self.assertFalse(result.compile_success)
        self.assertFalse(result.runtime_success)

    def test_runtime_exception_is_re(self) -> None:
        result = verify("raise RuntimeError('boom')\n", TESTS, sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.RE)
        self.assertTrue(result.compile_success)
        self.assertFalse(result.runtime_success)

    def test_infinite_loop_is_tle(self) -> None:
        result = verify("while True:\n    pass\n", TESTS, sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.TLE)
        self.assertTrue(result.compile_success)
        self.assertTrue(result.timeout)

    def test_wrong_output_is_wa(self) -> None:
        result = verify("print(6)\n", TESTS, sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.WA)
        self.assertTrue(result.compile_success)
        self.assertTrue(result.runtime_success)
        self.assertEqual(result.passed, 0)

    def test_fenced_python_code_is_extracted(self) -> None:
        response = "Here is the solution:\n```python\nprint(sum(map(int, input().split())))\n```"
        result = verify(response, TESTS, sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.AC)
        self.assertEqual(result.extraction_strategy, "fenced:python")

    def test_trailing_whitespace_is_normalized(self) -> None:
        result = verify("print('5   ')\nprint()\n", [{"input": "", "output": "5\n"}], sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.AC)

    def test_internal_whitespace_is_not_normalized_away(self) -> None:
        result = verify("print('1 2')\n", [{"input": "", "output": "1\n2\n"}], sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.WA)

    def test_output_limit_is_re(self) -> None:
        config = SandboxConfig(wall_time_seconds=0.3, cpu_time_seconds=1, memory_mb=256, output_limit_bytes=100)
        result = verify("print('x' * 1000)\n", [{"input": "", "output": ""}], sandbox_config=config)
        self.assertEqual(result.status, VerifierStatus.RE)
        self.assertEqual(result.test_results[0].error_type, "OutputLimitExceeded")

    def test_network_is_disabled(self) -> None:
        result = verify("import socket\nsocket.socket()\n", [{"input": "", "output": ""}], sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.RE)
        self.assertIn("operation disabled", result.test_results[0].error_message or "")

    def test_process_creation_is_disabled(self) -> None:
        result = verify(
            "import subprocess\nsubprocess.Popen(['python', '-c', 'print(1)'])\n",
            [{"input": "", "output": ""}],
            sandbox_config=self.sandbox_config(),
        )
        self.assertEqual(result.status, VerifierStatus.RE)
        self.assertIn("operation disabled", result.test_results[0].error_message or "")

    def test_filesystem_is_restricted(self) -> None:
        result = verify("open('/etc/passwd').read()\n", [{"input": "", "output": ""}], sandbox_config=self.sandbox_config())
        self.assertEqual(result.status, VerifierStatus.RE)
        self.assertIn("filesystem restricted", result.test_results[0].error_message or "")


if __name__ == "__main__":
    unittest.main()
