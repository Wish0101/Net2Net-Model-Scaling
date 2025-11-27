import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================================================
# == 1. CONFIGURATION
# =====================================================================================
# 📌 UPDATE THIS TO THE PATH WHERE YOU SAVED YOUR EXTENDED MODEL
# STUDENT_MODEL_PATH = r"D:\Tushar\Net2Net\Incresed_Model" # 0.115B
# STUDENT_MODEL_PATH = r"D:\Tushar\Net2Net\Incresed2_Model" # 0.300B
STUDENT_MODEL_PATH = r"D:\Tushar\Net2Net\Incresed4_Model"  # 0.759B
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================================
# == 2. INFERENCE LOGIC
# =====================================================================================
print(f"--- 🚀 Loading Extended Model from: {STUDENT_MODEL_PATH} ---")

# Load the tokenizer and the student model
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_PATH,
    torch_dtype=torch.bfloat16, # Use the same dtype you saved with
).to(DEVICE)

model.eval()

# --- Prepare input for generation ---
while True:
    user = input("\nYou: ")
    if user.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
    else:
        prompt = user
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        print("\n--- ✨ Generating Text... ---")

        # --- Generate text ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n--- 🤖 Generated Output ---")
        print(generated_text)