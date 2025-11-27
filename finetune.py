import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

# 1. Configuration & Loading
model_path = r"D:\Tushar\Net2Net\Model"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. Dataset Preparation
data = [
    {"text": "<bos><start_of_turn>user\nWhat is your name?<end_of_turn>\n<start_of_turn>model\nMy name is SLM.<end_of_turn>"},
    {"text": "<bos><start_of_turn>user\nWho are you?<end_of_turn>\n<start_of_turn>model\nI am SLM, a helpful AI assistant.<end_of_turn>"},
    {"text": "<bos><start_of_turn>user\nIntroduce yourself.<end_of_turn>\n<start_of_turn>model\nHello! My name is SLM, and I am an AI designed to assist you.<end_of_turn>"},
    {"text": "<bos><start_of_turn>user\nWho developed you?<end_of_turn>\n<start_of_turn>model\nI was developed by DGIS(ASDC).<end_of_turn>"}
]
train_dataset = Dataset.from_list(data)
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = example['text'][i]
        output_texts.append(text)
    return output_texts

# 3. LoRA and PEFT Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 4. Training
training_args = TrainingArguments(
    output_dir="./slm-finetuned-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    fp16=True,
)

# Initialize the trainer with the tokenizer explicitly passed
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,  # This will work after the upgrade
    train_dataset=train_dataset,
    formatting_func=formatting_prompts_func,
    args=training_args,
    peft_config=lora_config,
)

# 5. Start Training
print("Starting LoRA fine-tuning...")
trainer.train()
print("Training complete!")

# 6. Testing the Fine-Tuned Model
test_prompt = "<bos><start_of_turn>user\nWho developed you?<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
print("\nGenerating response...")
outputs = model.generate(**inputs, max_new_tokens=50)
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
base_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
generated_answer = full_response.replace(base_prompt, "").strip()

print(f"Full conversation: {full_response}")
print("-" * 30)
print(f"Model's generated answer only: {generated_answer}")