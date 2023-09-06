# Import required classes from the transformers library
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set the padding token to be the same as the End Of Sentence (eos) token
# This is necessary because GPT-2 doesn't have a default padding token
tokenizer.pad_token = tokenizer.eos_token

# Initialize the GPT-2 model for causal language modeling
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define the input text for the model
input_text = "who are you?"  # Sample input

# Tokenize the input text and prepare it for the model
# padding=True ensures that the input is padded to the model's expected input size
# truncation=True will truncate the input if it exceeds the model's maximum input size
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Generate text based on the input using the model
# max_length=50 limits the length of the generated text to 50 tokens
output = model.generate(**input_ids, max_length=50)

# Decode the generated text back into a readable string
# We skip the input text part in the output by slicing the string from len(input_text)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(input_text):]

# Print the generated text
print(output_text)