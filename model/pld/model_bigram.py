import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Configuration ---
MODEL_NAME = "/gemini/user/shared/models/vicuna-7b-v1.3" # Or a larger/quantized model
BATCH_SIZE = 100  # Adjust based on GPU memory
DEVICE = 1 if torch.cuda.is_available() else -1 # Use GPU if available

# --- 1. Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
# For quantized models, you might need AutoGPTQ or other specific loaders
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Set pad token if not present

# Use pipeline for easy batching (can also use model.generate directly)
# Specify device for GPU acceleration
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=DEVICE)
print("Model loaded.")

space_id = tokenizer.encode(" ", add_special_tokens=False)
space_id = space_id[0] # if isinstance(space_id, list) else space_id

bigram = [space_id for _ in range(tokenizer.vocab_size)]

# Get all valid token IDs (excluding special tokens)
valid_token_ids = [
    token_id for token_id in tokenizer.vocab.values() 
    if token_id != tokenizer.pad_token_id and token_id != tokenizer.eos_token_id
]


# Process tokens in batches
for i in tqdm(range(0, len(valid_token_ids), BATCH_SIZE)):
    batch_token_ids = valid_token_ids[i:i + BATCH_SIZE]
    batch_prompts = []
    
    # Prepare batch of prompts
    for token_id in batch_token_ids:
        try:
            prompt = tokenizer.decode(token_id)
            batch_prompts.append(prompt)
        except Exception as e:
            print(f"Error decoding token {token_id}: {e}")
            batch_prompts.append("")  # Add placeholder to maintain index alignment
    
    # Skip empty batch
    if not batch_prompts:
        continue
        
    # Generate next tokens for all prompts in batch
    try:
        outputs = generator(
            batch_prompts,
            max_length=2,  # Just need the prompt + one token
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        
        # Process results
        for j, (token_id, output) in enumerate(zip(batch_token_ids, outputs)):
            if not output:
                continue
                
            generated_text = output[0]['generated_text']
            prompt = batch_prompts[j]
            
            # Get the token_id of the next token (if it exists)
            if len(generated_text) > len(prompt):
                next_chars = generated_text[len(prompt):]
                next_token_ids = tokenizer.encode(next_chars, add_special_tokens=False)
                if next_token_ids:
                    bigram[token_id] = next_token_ids[0]
    except Exception as e:
        print(f"Error processing batch: {e}")

# Save the bigram mapping
with open("bigram_mapping.json", "w") as f:
    json.dump(bigram, f)