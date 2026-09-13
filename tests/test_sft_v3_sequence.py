from __future__ import annotations

import unittest

from eval.sft_v3_sequence import (
    build_sft_sequence_v3,
    canonical_sft_target,
    shifted_loss_positions,
)
from sft.train_v3 import DataCollatorForResponseOnlySftV3


class CharTokenizer:
    eos_token_id = 3
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        ids = [ord(char) + 10 for char in text]
        output: dict[str, object] = {"input_ids": ids}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output

    def decode(self, ids: list[int], **_: object) -> str:
        return "".join(chr(token - 10) for token in ids if token != self.eos_token_id)


class SftV3SequenceTest(unittest.TestCase):
    def config(self) -> dict[str, object]:
        return {"prompt": {"template": "### Problem\n{problem}\n\n### Solution\n"}}

    def test_canonical_target_keeps_one_complete_r1_response(self) -> None:
        response = "<think>done</think>\n```python\nprint(1)\n```\n"
        target = canonical_sft_target(self.config(), "Print one", response)

        self.assertEqual(target, "### Problem\nPrint one\n\n### Solution\n" + response)
        self.assertEqual(target.count("print(1)"), 1)

    def test_natural_full_fit_supervises_eos_and_masks_prompt(self) -> None:
        tokenizer = CharTokenizer()
        prompt = "### Problem\nx\n\n### Solution\n"
        response = "reason\nprint(1)\n"
        example = build_sft_sequence_v3(
            tokenizer,
            prompt=prompt,
            r1_generation=response,
            reference_code="print(1)\n",
            max_sequence_length=100,
            max_response_tokens_including_eos=4096,
        )

        self.assertFalse(example.dropped)
        self.assertEqual(tokenizer.decode(example.input_ids), prompt + response)
        self.assertEqual(example.input_ids[-1], tokenizer.eos_token_id)
        self.assertEqual(example.labels[-1], tokenizer.eos_token_id)
        self.assertTrue(all(label == -100 for label in example.labels[: example.prompt_tokens]))
        loss_positions = shifted_loss_positions(example.labels)
        self.assertTrue(loss_positions)
        self.assertTrue(all(index >= example.prompt_tokens for index in loss_positions))
        self.assertIn(len(example.labels) - 1, loss_positions)
        self.assertTrue(example.eos_supervised)
        self.assertFalse(example.artificial_truncation)
        self.assertFalse(example.reference_code_appended)

    def test_natural_response_over_budget_is_dropped_without_truncation(self) -> None:
        tokenizer = CharTokenizer()
        response = "reasoning\nprint(1)\n"
        example = build_sft_sequence_v3(
            tokenizer,
            prompt="P",
            r1_generation=response,
            reference_code="print(1)\n",
            max_sequence_length=100,
            max_response_tokens_including_eos=len(response),
        )

        self.assertTrue(example.dropped)
        self.assertEqual(example.drop_reason, "natural_response_over_budget")
        self.assertFalse(example.artificial_truncation)
        self.assertFalse(example.reference_code_appended)
        self.assertEqual(example.input_ids, [])

    def test_unclosed_think_is_dropped_even_when_code_is_locatable(self) -> None:
        tokenizer = CharTokenizer()
        example = build_sft_sequence_v3(
            tokenizer,
            prompt="P",
            r1_generation="<think>unfinished\nprint(1)\n",
            reference_code="print(1)\n",
            max_sequence_length=100,
            max_response_tokens_including_eos=4096,
        )

        self.assertTrue(example.dropped)
        self.assertEqual(example.drop_reason, "unclosed_or_late_think")
        self.assertTrue(example.code_locatable)
        self.assertFalse(example.has_matching_closed_think)

    def test_complete_fenced_response_is_retained_without_reconstruction(self) -> None:
        tokenizer = CharTokenizer()
        response = "<think>done</think>\n```python\nprint(1)\n```\n"
        example = build_sft_sequence_v3(
            tokenizer,
            prompt="P",
            r1_generation=response,
            reference_code="print(1)\n",
            max_sequence_length=100,
            max_response_tokens_including_eos=4096,
        )

        decoded_response = tokenizer.decode(example.input_ids[example.prompt_tokens :])
        self.assertFalse(example.dropped)
        self.assertEqual(decoded_response, response)
        self.assertEqual(decoded_response.count("print(1)"), 1)
        self.assertIn("</think>", decoded_response)
        self.assertIn("```python\n", decoded_response)
        self.assertTrue(decoded_response.endswith("```\n"))
        self.assertTrue(example.final_code_preserved)

    def test_total_sequence_over_budget_is_dropped(self) -> None:
        tokenizer = CharTokenizer()
        example = build_sft_sequence_v3(
            tokenizer,
            prompt="P" * 20,
            r1_generation="print(1)\n",
            reference_code="print(1)\n",
            max_sequence_length=10,
            max_response_tokens_including_eos=4096,
        )

        self.assertTrue(example.dropped)
        self.assertEqual(example.drop_reason, "prompt_plus_response_over_sequence_budget")

    def test_collator_masks_padding_but_keeps_eos_supervised(self) -> None:
        import torch

        collator = DataCollatorForResponseOnlySftV3(pad_token_id=0)
        batch = collator(
            [
                {"input_ids": [10, 11, 12, 3], "labels": [-100, -100, 12, 3]},
                {"input_ids": [10, 11, 3], "labels": [-100, -100, 3]},
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[10, 11, 12, 3], [10, 11, 3, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 12, 3], [-100, -100, 3, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertTrue(torch.is_tensor(batch["labels"]))


if __name__ == "__main__":
    unittest.main()
