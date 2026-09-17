from __future__ import annotations

import copy
import unittest
from pathlib import Path

from data.config import load_config
from sft.v4_data import controlled_config_diff


class SftV4ControlTest(unittest.TestCase):
    def base_configs(self):
        v3 = {
            "model": {"revision": "same"},
            "prompt": {"template": "same"},
            "sft_sequence": {"truncation_policy": "none", "max_sequence_length": 8192},
            "lora": {"r": 16, "alpha": 32},
            "training": {
                "seed": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "num_train_epochs": 1,
                "learning_rate": 0.0001,
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "logging_steps": 10,
                "save_strategy": "epoch",
                "bf16": True,
                "gradient_checkpointing": True,
                "optim": "adamw_torch",
                "report_to": "none",
                "output_dir": "v3",
            },
        }
        v4 = copy.deepcopy(v3)
        v4["training"]["learning_rate"] = 0.00005
        v4["training"]["output_dir"] = "v4"
        return v4, v3

    def test_only_learning_rate_and_output_path_change(self) -> None:
        v4, v3 = self.base_configs()
        diff = controlled_config_diff(v4, v3)
        self.assertTrue(diff["only_training_hyperparameter_change"])

    def test_rank_change_fails_controlled_comparison(self) -> None:
        v4, v3 = self.base_configs()
        v4["lora"]["r"] = 32
        diff = controlled_config_diff(v4, v3)
        self.assertFalse(diff["only_training_hyperparameter_change"])

    def test_epoch_change_fails_controlled_comparison(self) -> None:
        v4, v3 = self.base_configs()
        v4["training"]["num_train_epochs"] = 2
        diff = controlled_config_diff(v4, v3)
        self.assertFalse(diff["only_training_hyperparameter_change"])

    def test_frozen_candidate_and_lr_values(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "sft_v4.yaml")
        frozen = config["frozen_v3_candidate"]
        self.assertEqual(int(frozen["candidate_count"]), 726)
        self.assertEqual(
            frozen["dataset_hash"],
            "836b2569df27b6bb36cb8aa68e69335a5a6c9ffbf3ab959b2965bf3be045eb67",
        )
        self.assertEqual(
            frozen["manifest_hash"],
            "f752de76cfcb0ac6f24b082965fcb41762e7d8b8e772bebd70d8bf825dd58a4a",
        )
        self.assertEqual(float(config["training"]["learning_rate"]), 0.00005)
        self.assertEqual(int(config["training"]["num_train_epochs"]), 1)


if __name__ == "__main__":
    unittest.main()
