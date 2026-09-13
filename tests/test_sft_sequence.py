from __future__ import annotations

import unittest

from eval.sft_v2_sequence import (
    build_sft_sequence_v2,
    canonical_sft_target,
    format_response_diagnostics,
    locate_reference_code,
    response_only_labels,
    shifted_loss_positions,
)
from sft.train_v2 import DataCollatorForResponseOnlySftV2


class CharTokenizer:
    eos_token_id = 3

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        self.last_text = text
        ids = [ord(char) + 10 for char in text]
        output: dict[str, object] = {"input_ids": ids}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output

    def decode(self, ids: list[int], **_: object) -> str:
        return "".join(chr(token - 10) for token in ids if token != self.eos_token_id)


class SftSequenceTest(unittest.TestCase):
    def config(self) -> dict[str, object]:
        return {"prompt": {"template": "### Problem\n{problem}\n\n### Solution\n"}}

    def test_canonical_target_uses_r1_generation_once(self) -> None:
        target = canonical_sft_target(self.config(), "Add two numbers", "Think.\nprint(a + b)\n")

        self.assertEqual(target, "### Problem\nAdd two numbers\n\n### Solution\nThink.\nprint(a + b)\n")
        self.assertEqual(target.count("print(a + b)"), 1)

    def test_locate_reference_code_supports_whitespace_normalization_only(self) -> None:
        response = "Reason first.\n```python\nprint(1)   \nprint(2)\n```\n"
        location = locate_reference_code(response, "print(1)\nprint(2)")

        self.assertIsNotNone(location)
        self.assertEqual(location.method, "normalized_whitespace")

    def test_full_fit_keeps_complete_response_and_masks_prompt(self) -> None:
        tokenizer = CharTokenizer()
        prompt = "### Problem\nx\n\n### Solution\n"
        response = "reason\nprint(1)\n"
        example = build_sft_sequence_v2(
            tokenizer,
            prompt=prompt,
            r1_generation=response,
            reference_code="print(1)\n",
            max_sequence_length=100,
            max_response_tokens_including_eos=4096,
        )

        self.assertFalse(example.dropped)
        self.assertTrue(example.full_fit)
        self.assertEqual(tokenizer.decode(example.input_ids), prompt + response)
        self.assertTrue(all(label == -100 for label in example.labels[: example.prompt_tokens]))
        self.assertTrue(all(label != -100 for label in example.labels[example.prompt_tokens :]))
        self.assertEqual(example.input_ids[-1], tokenizer.eos_token_id)
        self.assertEqual(example.labels[-1], tokenizer.eos_token_id)
        self.assertTrue(all(index >= example.prompt_tokens for index in shifted_loss_positions(example.labels)))
        self.assertIn(len(example.labels) - 1, shifted_loss_positions(example.labels))

    def test_overflow_truncates_reasoning_before_code_and_preserves_code(self) -> None:
        tokenizer = CharTokenizer()
        prompt = "P" * 5
        reasoning = "r" * 20
        code = "print(1)\n"
        example = build_sft_sequence_v2(
            tokenizer,
            prompt=prompt,
            r1_generation=reasoning + code,
            reference_code=code,
            max_sequence_length=5 + 8 + len(code) + 1,
            max_response_tokens_including_eos=8 + len(code) + 1,
        )

        decoded = tokenizer.decode(example.input_ids)
        self.assertFalse(example.dropped)
        self.assertTrue(example.reasoning_truncated)
        self.assertEqual(decoded, prompt + "r" * 8 + code)
        self.assertEqual(example.input_ids[-1], tokenizer.eos_token_id)
        self.assertEqual(example.labels[-1], tokenizer.eos_token_id)

    def test_think_overflow_keeps_close_and_both_fences(self) -> None:
        tokenizer = CharTokenizer()
        code = "print(1)\n"
        response = "<think>\n" + ("reasoning " * 20) + "\n</think>\n```python\n" + code + "```\n"
        example = build_sft_sequence_v2(
            tokenizer,
            prompt="P",
            r1_generation=response,
            reference_code=code,
            max_sequence_length=64,
            max_response_tokens_including_eos=48,
        )

        decoded = tokenizer.decode(example.input_ids[example.prompt_tokens :])
        self.assertFalse(example.dropped)
        self.assertTrue(example.reasoning_truncated)
        self.assertTrue(decoded.startswith("<think>"))
        self.assertIn("</think>", decoded)
        self.assertIn("```python\n", decoded)
        self.assertTrue(decoded.endswith(code + "```\n"))
        self.assertEqual(decoded.count(code), 1)
        self.assertEqual(example.labels[-1], tokenizer.eos_token_id)

    def test_overflow_unlocatable_code_is_dropped(self) -> None:
        tokenizer = CharTokenizer()
        example = build_sft_sequence_v2(
            tokenizer,
            prompt="P" * 5,
            r1_generation="r" * 20 + "print(1)\n",
            reference_code="print(2)\n",
            max_sequence_length=12,
            max_response_tokens_including_eos=10,
        )

        self.assertTrue(example.dropped)
        self.assertEqual(example.drop_reason, "unlocatable_final_code")

    def test_response_only_labels_include_eos_and_exclude_padding(self) -> None:
        labels = response_only_labels(
            [10, 11, 12, 3, 0],
            prompt_tokens=2,
            response_tokens_including_eos=2,
            pad_token_id=0,
        )

        self.assertEqual(labels, [-100, -100, 12, 3, -100])
        self.assertEqual(shifted_loss_positions(labels), [2, 3])

    def test_format_diagnostics_reject_unclosed_think_even_when_extraction_succeeds(self) -> None:
        diagnostics = format_response_diagnostics("<think>unfinished\nprint(1)\n", True)

        self.assertTrue(diagnostics["starts_with_think"])
        self.assertTrue(diagnostics["unclosed_think"])
        self.assertFalse(diagnostics["valid_reasoning_to_code_transition"])
        self.assertFalse(diagnostics["format_valid"])

    def test_collator_masks_padding_but_keeps_eos_supervised(self) -> None:
        import torch

        collator = DataCollatorForResponseOnlySftV2(pad_token_id=0)
        batch = collator(
            [
                {"input_ids": [10, 11, 12, 3], "labels": [-100, -100, 12, 3]},
                {"input_ids": [10, 11, 3], "labels": [-100, -100, 3]},
            ]
        )

        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 12, 3], [-100, -100, 3, -100]])
        self.assertEqual(batch["input_ids"].tolist(), [[10, 11, 12, 3], [10, 11, 3, 0]])
        self.assertTrue(torch.is_tensor(batch["labels"]))


if __name__ == "__main__":
    unittest.main()
