# main.py - Line 1
from transformers import AutoModelForCausalLM, AutoTokenizer
# main.py - Line 3
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
# main.py - Line 6-12
input_text = "Hello, how are you?"  # Sample input
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

