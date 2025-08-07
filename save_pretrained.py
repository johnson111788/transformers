import torch
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    GptVOssConfig, GptVOssForConditionalGeneration
)

from accelerate import init_empty_weights
import pdb

INTERNVL_REPO = "/home/efs/hardychen/models/InternVL3-1B-hf"
GPTOSS_REPO   = "openai/gpt-oss-20b"

# 1. Load the two source checkpoints once
internvl_model = AutoModel.from_pretrained(
    INTERNVL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16
)
gptoss_model = AutoModelForCausalLM.from_pretrained(
    GPTOSS_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16
)
import ipdb;ipdb.set_trace()
# 2. Build a combined config
proj_dim  = internvl_model.config.text_config.hidden_size
comp_cfg  = GptVOssConfig(
    internvl_config       = internvl_model.config,
    text_config           = gptoss_model.config,
    projector_hidden_size = proj_dim,
)

# 3. Instantiate the wrapper **without allocating real weights**
with init_empty_weights():           # `no_init_weights()` on v4.39-
    comp_model = GptVOssForConditionalGeneration(comp_cfg)

# 4. Load the real parameters
state = {}
state.update({f"model.vision_language_model.{k}": v
              for k, v in internvl_model.state_dict().items()})
state.update({f"model.oss_language_model.{k}": v
              for k, v in gptoss_model.state_dict().items()})

missing, unexpected = comp_model.load_state_dict(state, strict=False, assign=True)
print("missing keys :", missing)
print("unexpected   :", unexpected)


################

# NOTE: 
# We use 
# self.language_model = AutoModelForCausalLM.from_config(config.text_config)
# in modeling_gpt_v_oss.py so that we can obtain the weight of lm_head.
# We should use 
# self.language_model = AutoModel.from_config(config.text_config)
# afterwards.

####### deal with lm_head #######
# point the outer head at the already-loaded weight
comp_model.lm_head = comp_model.model.oss_language_model.lm_head

# copy the weigth of inner lm_head to the outer one
comp_model.lm_head.weight.data = comp_model.lm_head.weight.data.copy_(
    comp_model.model.oss_language_model.lm_head.weight.data
)

# keep the base model only
comp_model.model.oss_language_model = comp_model.model.oss_language_model.model
####### deal with lm_head #######


####### deal with projector #######
import math, torch, torch.nn as nn

def init_linear(linear: nn.Linear):
    # create real tensors on CPU
    w = torch.empty(linear.weight.shape, dtype=torch.bfloat16, device="cpu")
    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    linear.weight = nn.Parameter(w)

    if linear.bias is not None:
        b = torch.zeros_like(linear.bias, device="cpu")
        linear.bias = nn.Parameter(b)

# materialise both layers
init_linear(comp_model.model.projector.fc1)
init_linear(comp_model.model.projector.fc2)
####### deal with projector #######
# pdb.set_trace()


# 5. (optional) dispatch across GPUs / move to bfloat16, etc.
# from accelerate import infer_auto_device_map, dispatch_model
# device_map = infer_auto_device_map(comp_model, max_memory={0:"30GiB",1:"30GiB"})
# comp_model = dispatch_model(comp_model, device_map=device_map)

# 6. Save
save_dir = "/home/ychou11/projects/11-OpenAI/gpt_v_oss"
comp_model.save_pretrained(save_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(INTERNVL_REPO, trust_remote_code=True).save_pretrained(save_dir)
AutoProcessor.from_pretrained(INTERNVL_REPO, trust_remote_code=True).save_pretrained(save_dir)
print("✅ InternVL + GPT-OSS merged and saved to", save_dir)
