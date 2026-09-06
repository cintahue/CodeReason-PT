from __future__ import annotations

import unittest

from verifier.qualify_phase1 import (
    _wa_diagnostic_from_runs,
    classify_protocol,
    final_candidate_flags,
)
from verifier.result import VerificationResult, VerifierStatus


class VerifierQualificationTest(unittest.TestCase):
    def test_protocol_stdio_requires_empty_fn_name_and_starter(self) -> None:
        row = {"input_output": '{"inputs":["1\\n"],"outputs":["1\\n"],"fn_name":""}', "starter_code": ""}
        protocol = classify_protocol(row)
        self.assertEqual(protocol["protocol"], "stdio")
        self.assertTrue(protocol["stdio_eligible"])

    def test_protocol_call_based_when_fn_name_present(self) -> None:
        row = {"input_output": '{"inputs":[[1]],"outputs":[1],"fn_name":"solve"}', "starter_code": ""}
        protocol = classify_protocol(row)
        self.assertEqual(protocol["protocol"], "call_based")
        self.assertFalse(protocol["stdio_eligible"])

    def test_protocol_call_based_when_starter_present(self) -> None:
        row = {"input_output": '{"inputs":["1\\n"],"outputs":["1\\n"]}', "starter_code": "class Solution: pass"}
        protocol = classify_protocol(row)
        self.assertEqual(protocol["protocol"], "call_based")
        self.assertFalse(protocol["stdio_eligible"])

    def test_final_candidate_definitions(self) -> None:
        base = {
            "stdio_eligible": True,
            "source_reference_verified": True,
            "ocr_solution_verified": False,
        }
        self.assertFalse(final_candidate_flags({**base, "dataset_split": "sft"})["sft_candidate"])
        self.assertTrue(final_candidate_flags({**base, "dataset_split": "pt"})["pt_candidate"])
        self.assertTrue(final_candidate_flags({**base, "dataset_split": "dev"})["dev_candidate"])

    def test_wa_diagnostic_detects_token_equal_mismatch(self) -> None:
        result = VerificationResult(
            status=VerifierStatus.WA,
            compile_success=True,
            runtime_success=True,
            timeout=False,
            passed=0,
            total=1,
            pass_rate=0.0,
            exit_code=0,
            runtime_ms=0,
            stdout_size=4,
            stderr_size=0,
            extraction_strategy="source_solution_1",
            sandbox_backend="local",
            normalization_policy=[],
            sandbox={},
        )
        diagnostic = _wa_diagnostic_from_runs(
            result=result,
            raw_runs=[{"stdout": "1  2\n", "exit_code": 0}],
            tests=[{"output": "1 2\n"}],
            prompt="Print the answer.",
        )
        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic["category"], "whitespace_tokenization_mismatch")


if __name__ == "__main__":
    unittest.main()
