import torch
from transformers import AutoModelForCausalLM
import os

# =====================================================================================
# == 1. CONFIGURATION
# =====================================================================================
# 📌 Ensure these paths are correct
TEACHER_MODEL_PATH = r"D:\Tushar\Net2Net\Model"
STUDENT_MODEL_PATH = r"D:\Tushar\Net2Net\Incresed4_Model"

# =====================================================================================
# == 2. PARAMETER CALCULATION FUNCTION
# =====================================================================================
def get_model_parameters(model_name: str, path: str):
    """Loads a model and prints its parameter count."""
    print("="*50)
    print(f"Analyzing: {model_name}")
    print(f"Path: {path}")
    print("="*50)
    
    try:
        # Load the model onto the CPU for this calculation to save VRAM
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")
        
        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate the number of trainable parameters (all should be trainable here)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        params_in_billions = total_params / 1_000_000_000
        
        print(f" -> Total Parameters:     {total_params:,}")
        print(f" -> Trainable Parameters: {trainable_params:,}")
        print(f" -> Size in Billions:     {params_in_billions:.3f}B")
        print("\n")
        
        return total_params
        
    except Exception as e:
        print(f"Could not load or analyze model at {path}. Error: {e}")
        return 0

# =====================================================================================
# == 3. EXECUTION AND COMPARISON
# =====================================================================================
if __name__ == "__main__":
    teacher_params = get_model_parameters("Original Teacher Model", TEACHER_MODEL_PATH)
    student_params = get_model_parameters("New Student (Widened) Model", STUDENT_MODEL_PATH)
    
    if teacher_params > 0 and student_params > 0:
        difference = student_params - teacher_params
        print("="*50)
        print("Comparison Summary")
        print("="*50)
        print(f" -> Parameters Added: {difference:,}")
        print(f" -> Increase (Billions): {(difference / 1_000_000_000):.3f}B")