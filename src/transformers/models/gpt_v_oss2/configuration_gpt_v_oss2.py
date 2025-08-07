"""GPT-VOss model configuration."""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig

logger = logging.get_logger(__name__)


class GptVOss2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GptVOss2Model`]. It is used to instantiate a
    GPT-VOss model according to the specified arguments. An example of such a model is
    [`your-org/gpt-v-oss-20b`](https://huggingface.co/your-org/gpt-v-oss-20b).
    """

    model_type = "gpt_v_oss"
    sub_configs = {"internvl_config": AutoConfig, "text_config": AutoConfig}

    def __init__(
        self,
        internvl_config=None,
        text_config=None,
        projector_hidden_size: int | None = None,
        **kwargs,
    ):
        if isinstance(internvl_config, dict):
            internvl_config["model_type"] = internvl_config.get("model_type", "internvl")
            internvl_config = CONFIG_MAPPING[internvl_config["model_type"]](**internvl_config)
        elif internvl_config is None:
            internvl_config = CONFIG_MAPPING["internvl"]()
        self.internvl_config = internvl_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "gpt_oss")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["gpt_oss"]()
        self.text_config = text_config

        self.projector_hidden_size = (
            projector_hidden_size
            if projector_hidden_size is not None
            else self.text_config.hidden_size
        )

        super().__init__(**kwargs)


__all__ = ["GptVOss2Config"]
