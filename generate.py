from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
import torch
from consts import PROMPT_FOR_GENERATION_FORMAT
from instruct_pipeline import InstructionTextGenerationPipeline

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(bnb_config.load_in_4bit)

model_id = ""
model_id = "tiiuae/falcon-40b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    return_dict=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    # context_length=1024,
    # rescale_every=0,
    trust_remote_code=True,
    device_map={"":0}
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = InstructionTextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    top_p=0.92,
    top_k=50,
    temperature=1.0,
    do_sample=False
)
instruction = "Write me the steps to make a peanut butter and jelly sandwich"
prompt = PROMPT_FOR_GENERATION_FORMAT.format(
    instruction=instruction,
)

class IsBork(LogitsProcessor):
    def __call__(self, input_ids, scores):
        print(scores)
        return scores
    
# prompt = f"Bob: {instruction}\nAlice: "
prompt = str(prompt)
inputs = tokenizer(prompt, return_tensors="pt")

input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
input_ids, attention_mask = input_ids.to("cuda"), attention_mask.to("cuda")

generated_sequence = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.pad_token_id,
    top_p=0.92,
    top_k=50,
    temperature=1.0,
    do_sample=False,
    max_new_tokens=512
)

generated_sequence = tokenizer.decode(generated_sequence[0], skip_special_tokens=False)
print(generated_sequence)
