from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor
import torch
from consts import PROMPT_FOR_GENERATION_FORMAT
from instruct_pipeline import InstructionTextGenerationPipeline

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True
# )

model = AutoModelForCausalLM.from_pretrained(
    "RWKV/rwkv-raven-14b",
    return_dict=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    rescale_every=0,
).cuda()

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-raven-14b")

pipeline = InstructionTextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    top_p=0.92,
    top_k=50,
    temperature=1.0,
    do_sample=True
)

prompt = PROMPT_FOR_GENERATION_FORMAT.format(
    instruction="Write me the steps to make a peanut butter and jelly sandwich.",
)

prompt = str(prompt)

gen = pipeline(prompt, max_new_tokens=512)

print(gen)