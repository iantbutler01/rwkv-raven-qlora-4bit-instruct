import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "./output/dist"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.float16,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

merged_model = model.base_model.merge_and_unload()
model_state = merged_model.state_dict()
tokenizer.save_pretrained(f"./output/merged")
merged_model.save_pretrained(f"./output/merged", state_dict=model_state)
