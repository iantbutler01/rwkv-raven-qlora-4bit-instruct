from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True
)

merged_model = "./output/merged"

model = AutoModelForCausalLM.from_pretrained(
    merged_model,
    return_dict=True,
    device_map={"":0}
)

tokenizer = AutoTokenizer.from_pretrained(merged_model)

model.push_to_hub("RWKV-14bn-4bit-QLORA-MERGED-Dolly-Instruct-Tuned")
tokenizer.push_to_hub("RWKV-14bn-4bit-QLORA-MERGED-Dolly-Instruct-Tuned")
