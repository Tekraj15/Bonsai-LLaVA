# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

# Chat formatting. Qwen's chat template injects a Qwen-branded default system
# prompt when messages don't start with a system turn, so we always supply our own.
# Training and inference MUST use this same constant or the model sees a different
# prefix at test time than it was trained on.
SYSTEM_PROMPT = "You are a helpful visual assistant. Answer the user's questions about the image accurately and concisely."
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Mean and Std for SigLIP
# SigLIP usually handles this in the processor, but good to have if manual.
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
