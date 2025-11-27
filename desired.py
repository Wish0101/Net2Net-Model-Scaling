import torch
from transformers import AutoConfig
import math

# =====================================================================================
# == 1. CONFIGURATION
# =====================================================================================
# 📌 Set the path to your ORIGINAL model and your desired parameter increase
TEACHER_MODEL_PATH = "Net2Net/Model"
DESIRED_INCREASE_IN_BILLIONS = 0.35

# =====================================================================================
# == 2. CALCULATION
# =====================================================================================
print("--- 🧠 Loading teacher model configuration ---")
try:
    config = AutoConfig.from_pretrained(TEACHER_MODEL_PATH)

    old_intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    desired_increase_params = DESIRED_INCREASE_IN_BILLIONS * 1_000_000_000

    print(f"\nOriginal Model Config:")
    print(f"  - Hidden Size:         {hidden_size}")
    print(f"  - Intermediate Size:   {old_intermediate_size}")
    print(f"  - Number of Layers:    {num_layers}")

    # The formula for parameter increase is: num_layers * (2 * hidden_size * delta_intermediate)
    # This assumes bias=False, which is standard for Gemma.
    params_per_intermediate_dim = num_layers * 2 * hidden_size
    
    # Calculate how much we need to increase the intermediate dimension
    delta_intermediate_size = desired_increase_params / params_per_intermediate_dim
    new_intermediate_size_float = old_intermediate_size + delta_intermediate_size
    
    # --- ❗ KEY CHANGE ---
    # We now round UP to the nearest multiple to ensure we meet or exceed the target.
    multiple = 128
    new_intermediate_size_rounded_up = math.ceil(new_intermediate_size_float / multiple) * multiple

    print("\n" + "="*50)
    print("--- 📈 CALCULATION RESULTS (REVISED) ---")
    print(f"To add at least {DESIRED_INCREASE_IN_BILLIONS}B parameters, the intermediate size must increase by ~{int(delta_intermediate_size)}.")
    print(f"The suggested NEW_INTERMEDIATE_SIZE (rounded UP) is: {new_intermediate_size_rounded_up}")
    print("="*50)

except Exception as e:
    print(f"Error: Could not process the model configuration. {e}")