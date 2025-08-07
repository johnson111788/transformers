import unittest

import torch

from transformers import AutoConfig
from transformers.models.gpt_v_oss import (
    GptVOssConfig,
    GptVOssModel,
    GptVOssForConditionalGeneration,
)


class GptVOssModelTest(unittest.TestCase):
    def setUp(self):
        vision_config = {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "image_size": 14,
            "patch_size": 14,
        }
        text_backbone_config = {
            "model_type": "llama",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
        }
        internvl_config = AutoConfig.for_model(
            "internvl",
            vision_config=vision_config,
            text_config=text_backbone_config,
        )
        gpt_config = AutoConfig.for_model(
            "gpt_oss",
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            sliding_window=32,
            num_local_experts=1,
            num_experts_per_tok=1,
        )
        self.config = GptVOssConfig(
            internvl_config=internvl_config,
            text_config=gpt_config,
            projector_hidden_size=16,
        )

    def test_forward(self):
        model = GptVOssModel(self.config)
        input_ids = torch.randint(0, self.config.internvl_config.text_config.vocab_size, (1, 4))
        output = model(input_ids=input_ids)
        self.assertEqual(output.last_hidden_state.shape, (1, 4, self.config.text_config.hidden_size))

    def test_for_conditional_generation(self):
        model = GptVOssForConditionalGeneration(self.config)
        input_ids = torch.randint(0, self.config.internvl_config.text_config.vocab_size, (1, 4))
        labels = torch.randint(0, self.config.text_config.vocab_size, (1, 4))
        output = model(input_ids=input_ids, labels=labels)
        self.assertEqual(output.logits.shape, (1, 4, self.config.text_config.vocab_size))
        self.assertIsNotNone(output.loss)


if __name__ == "__main__":
    unittest.main()
