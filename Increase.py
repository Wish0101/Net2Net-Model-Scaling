import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaConfig
import os

# =====================================================================================
# == 1. CONFIGURATION
# =====================================================================================
TEACHER_MODEL_PATH = "Net2Net/Model"
STUDENT_MODEL_PATH = "Net2Net/Incresed_Model"
NEW_INTERMEDIATE_SIZE = 12800  # Must be > original size

# =====================================================================================
# == 2. CORE NET2WIDERNET FUNCTION (NOW HANDLES BIASES)
# =====================================================================================
def apply_net2wider(teacher_mlp, student_mlp):
    """
    Applies the Net2WiderNet transformation, now handling both weights and biases.
    """
    teacher_gate_proj = teacher_mlp.gate_proj
    teacher_up_proj = teacher_mlp.up_proj
    teacher_down_proj = teacher_mlp.down_proj

    student_gate_proj = student_mlp.gate_proj
    student_up_proj = student_mlp.up_proj
    student_down_proj = student_mlp.down_proj

    old_width = teacher_gate_proj.out_features
    new_width = student_gate_proj.out_features

    if new_width <= old_width:
        raise ValueError("New width must be greater than old width for Net2WiderNet.")

    # --- Create a replication map for the new neurons ---
    replication_map = np.random.randint(0, old_width, size=(new_width - old_width))

    # --- Widen the gate_proj and up_proj layers (weights and biases) ---
    for teacher_layer, student_layer in [
        (teacher_gate_proj, student_gate_proj),
        (teacher_up_proj, student_up_proj)
    ]:
        # Handle weights
        student_layer.weight.data[:old_width, :] = teacher_layer.weight.data.clone()
        student_layer.weight.data[old_width:, :] = teacher_layer.weight.data[replication_map, :].clone()

        # Handle biases if they exist
        if teacher_layer.bias is not None and student_layer.bias is not None:
            student_layer.bias.data[:old_width] = teacher_layer.bias.data.clone()
            student_layer.bias.data[old_width:] = teacher_layer.bias.data[replication_map].clone()

    # --- Handle the down_proj layer (weights and biases) ---
    # Copy/replicate weights
    student_down_proj.weight.data[:, :old_width] = teacher_down_proj.weight.data.clone()
    student_down_proj.weight.data[:, old_width:] = teacher_down_proj.weight.data[:, replication_map].clone()
    
    # Copy bias directly if it exists (output dimension doesn't change)
    if teacher_down_proj.bias is not None and student_down_proj.bias is not None:
        student_down_proj.bias.data = teacher_down_proj.bias.data.clone()

    # --- Scale the down_proj weights to preserve function ---
    scaling_vector = torch.ones(new_width, device=student_down_proj.weight.device)
    for i in range(old_width):
        count = 1 + np.count_nonzero(replication_map == i)
        if count > 1:
            scaling_vector[i] = count
    for i in range(len(replication_map)):
        original_neuron_idx = replication_map[i]
        new_neuron_idx = old_width + i
        scaling_vector[new_neuron_idx] = scaling_vector[original_neuron_idx]
    student_down_proj.weight.data /= scaling_vector.unsqueeze(0)


# =====================================================================================
# == 3. MAIN EXECUTION LOGIC
# =====================================================================================
if __name__ == "__main__":
    print("--- 🧠 Loading Teacher Model ---")
    
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    teacher_config = teacher_model.config
    old_intermediate_size = teacher_config.intermediate_size
    
    print(f"Original intermediate size: {old_intermediate_size}")
    print(f"New intermediate size: {NEW_INTERMEDIATE_SIZE}")
    
    # Use the teacher's config to ensure all params (like bias=True/False) are identical
    student_config = teacher_model.config
    student_config.intermediate_size = NEW_INTERMEDIATE_SIZE
    
    print("\n--- 🏗️ Creating Student Model Architecture ---")
    student_model = AutoModelForCausalLM.from_config(student_config).to(
        teacher_model.device, dtype=teacher_model.dtype
    )

    teacher_params = sum(p.numel() for p in teacher_model.parameters()) / 1_000_000
    student_params = sum(p.numel() for p in student_model.parameters()) / 1_000_000
    print(f"Teacher parameter count: {teacher_params:.2f}M")
    print(f"Student parameter count: {student_params:.2f}M")
    print(f"Parameters added: {student_params - teacher_params:.2f}M")
    
    print("\n--- 🧬 Transferring and Expanding Weights ---")
    
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = student_model.state_dict()
    state_dict_to_load = {}
    
    # Define all MLP keys to skip (both weights and biases)
    mlp_keys_to_skip = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    for name, param in teacher_state_dict.items():
        # Check if the current parameter belongs to any of the MLP layers we're manually handling
        if any(skip_key in name for skip_key in mlp_keys_to_skip):
            continue
        
        # If not an MLP key, and it exists in the student, add it for loading
        if name in student_state_dict and student_state_dict[name].shape == param.shape:
            state_dict_to_load[name] = param

    student_model.load_state_dict(state_dict_to_load, strict=False)
    print("  -> Copied all compatible, non-MLP weights.")

    num_layers = teacher_config.num_hidden_layers
    for i in range(num_layers):
        teacher_mlp = teacher_model.model.layers[i].mlp
        student_mlp = student_model.model.layers[i].mlp
        
        apply_net2wider(teacher_mlp, student_mlp)
        print(f"  -> Widened MLP for layer {i+1}/{num_layers}")

    print("\n--- ✅ Weight Transfer Complete ---")

    print("\n--- 🧐 Verifying Function Preservation ---")
    teacher_model.eval()
    student_model.eval()
    
    prompt = "The best thing about AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(teacher_model.device)
    
    with torch.no_grad():
        teacher_output = teacher_model(**inputs, output_hidden_states=True)
        student_output = student_model(**inputs, output_hidden_states=True)

    teacher_last_hidden = teacher_output.hidden_states[-1]
    student_last_hidden = student_output.hidden_states[-1]
    
    are_outputs_close = torch.allclose(teacher_last_hidden, student_last_hidden, atol=1e-3)
    print(f"Are model outputs (logits precursor) close? -> {are_outputs_close}")
    if not are_outputs_close:
        max_diff = (teacher_last_hidden - student_last_hidden).abs().max().item()
        print(f"Warning: Outputs are not identical. Max difference: {max_diff:.6f}")

    print(f"\n--- 💾 Saving Student Model to: {STUDENT_MODEL_PATH} ---")
    student_model.save_pretrained(STUDENT_MODEL_PATH)
    tokenizer.save_pretrained(STUDENT_MODEL_PATH)
    
    print("\n🎉 Process finished successfully!")