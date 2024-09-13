from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
model_id = "teddylee777/Llama-3-Open-Ko-8B-Instruct-teddynote-gguf"
filename = "Llama-3-Open-Ko-8B-Instruct-teddynote-gguf-unsloth.Q8_0.gguf"

# tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
# model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", # TypeError: BFloat16 is not supported on MPS
    device_map="auto",
    gguf_file=filename,
    # torch_dtype=torch.bfloat16, # 메모리 에러...
)

messages = [
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "피보나치 수열이 뭐야? 그리고 피보나치 수열에 대해 파이썬 코드를 짜줘볼래?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=1,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))