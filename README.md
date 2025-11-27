# Net2Net-Model-Scaling
A PyTorch implementation of Net2WiderNet for scaling up Transformer-based LLMs (e.g., Gemma, Llama). This toolkit allows you to widen MLP layers to increase parameter count while preserving model function, complete with scripts for calculation, validation, inference, and LoRA fine-tuning.

# Net2WiderNet LLM Expander 🧠📈

This repository contains a toolkit for implementing the **Net2WiderNet** technique on Transformer-based Large Language Models (LLMs). It allows you to take a pre-trained "Teacher" model and significantly increase its parameter count by widening the MLP (Feed Forward) intermediate layers to create a "Student" model.

Critically, this expansion preserves the original function of the model (function preservation), allowing you to scale up a smaller model and continue training (or fine-tuning) it without starting from scratch.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `Increase.py` | **Core Script.** Performs the Net2WiderNet expansion. It maps weights from the Teacher to the Student, replicates neurons, and scales the down-projection layers to preserve output logic. |
| `desired.py` | A utility to calculate the required `intermediate_size` to achieve a specific target parameter increase (e.g., +0.35B params). |
| `Param.py` | A diagnostic tool to compare the parameter counts of the Original (Teacher) vs. Expanded (Student) models. |
| `inference.py` | An interactive script to load the expanded model and chat with it to verify it is not broken. |
| `finetune.py` | A post-expansion training script using **LoRA** (Low-Rank Adaptation) and **4-bit quantization** to fine-tune the expanded model on new data. |

## 🚀 How It Works (Net2WiderNet)

The `Increase.py` script implements the Net2WiderNet algorithm for the MLP blocks of a Transformer (`gate_proj`, `up_proj`, `down_proj`):

1.  **Architecture Adjustment:** A new Student config is created with a larger `intermediate_size`.
2.  **Weight Copying:** Existing weights are copied directly from Teacher to Student.
3.  **Smart Replication:** To fill the new dimensions, random indices from the original layers are chosen to replicate weights into the new neurons.
4.  **Normalization/Scaling:** The `down_proj` layer weights are scaled inversely to the number of times a neuron was replicated. This ensures that the mathematical output of the Student model is nearly identical to the Teacher model at initialization.

## 🛠️ Requirements

Install the necessary dependencies. It is recommended to use a virtual environment.

```bash
pip install torch transformers peft bitsandbytes trl accelerate datasets