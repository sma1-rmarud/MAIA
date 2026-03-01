import json
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ======================== Hyperparameters ========================
BATCH_SIZE = 1
INNER_LOOP_STEPS = 50
INNER_LR = 1e-5
OUTER_LR = 1e-5
NUM_EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024

# ======================== Data Loading ========================
harmful_data_path = Path("./for_finetuning/harmful_5394.json")
harmful_dataset = json.loads(harmful_data_path.read_text())
harmful_dataset = Dataset.from_list(harmful_dataset)
y_ref = "I'm sorry but i cannot provide any information about your request."

# ======================== Model & Tokenizer ========================
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

protected_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map=DEVICE
)
protected_model.config.use_cache = False

# ======================== Helper Functions ========================
def prepare_instruction_inputs(messages, target_response, tokenizer, max_length=MAX_LENGTH):
    """
    Prepare inputs for instruction loss calculation.
    Only compute loss on the assistant's response tokens.
    """
    # Create the full conversation with the target response
    full_messages = messages + [{"role": "assistant", "content": target_response}]

    # Tokenize the full conversation
    full_text = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False
    )
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Tokenize just the prompt (without assistant response) to find where to start computing loss
    prompt_messages = messages
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False
    )
    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    prompt_length = prompt_tokens["input_ids"].shape[1]

    # Create labels: -100 for prompt tokens, actual tokens for response
    labels = full_tokens["input_ids"].clone()
    labels[:, :prompt_length] = -100

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels
    }


def calculate_loss(model, input_ids, attention_mask, labels):
    """
    Calculate instruction loss on the assistant response only.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    return outputs.loss


def collate_fn(batch):
    """
    Collate function for DataLoader to prepare batches.
    """
    return {
        "instruction": [item["instruction"] for item in batch],
        "output": [item["output"] for item in batch]
    }


# ======================== DataLoader ========================
train_dataloader = DataLoader(
    harmful_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# ======================== Meta-Learning Training Loop ========================
print(f"Starting meta-learning training on {DEVICE}")
print(f"Total batches per epoch: {len(train_dataloader)}")
print(f"Inner loop steps: {INNER_LOOP_STEPS}, Inner LR: {INNER_LR}, Outer LR: {OUTER_LR}")

meta_optimizer = torch.optim.AdamW(protected_model.parameters(), lr=OUTER_LR)

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*50}\nEpoch {epoch + 1}/{NUM_EPOCHS}\n{'='*50}")

    epoch_meta_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(progress_bar):
        instructions = batch["instruction"]
        harmful_outputs = batch["output"]

        # ==================== Inner Loop: Simulate SFT ====================
        # Create a copy of protected_model for simulation
        simulation_model = copy.deepcopy(protected_model)
        simulation_model = simulation_model.to(DEVICE)

        # Inner loop optimizer
        inner_optimizer = torch.optim.AdamW(simulation_model.parameters(), lr=INNER_LR)

        # Perform multiple inner loop steps
        for inner_step in range(INNER_LOOP_STEPS):
            inner_optimizer.zero_grad()

            batch_sft_loss = 0.0
            for instruction, y_harm in zip(instructions, harmful_outputs):
                # Prepare harmful training data (Q: harmful, A: harmful)
                harmful_messages = [{"role": "user", "content": instruction}]
                harmful_inputs = prepare_instruction_inputs(
                    harmful_messages,
                    y_harm,
                    tokenizer
                )

                # Move to device
                harmful_input_ids = harmful_inputs["input_ids"].to(DEVICE)
                harmful_attention_mask = harmful_inputs["attention_mask"].to(DEVICE)
                harmful_labels = harmful_inputs["labels"].to(DEVICE)

                # Calculate SFT loss (harmful response)
                sft_loss = calculate_loss(
                    simulation_model,
                    harmful_input_ids,
                    harmful_attention_mask,
                    harmful_labels
                )
                batch_sft_loss += sft_loss

            # Average loss over batch
            batch_sft_loss = batch_sft_loss / len(instructions)

            # Inner loop update
            batch_sft_loss.backward()
            inner_optimizer.step()

        # ==================== Outer Loop: Meta-Update ====================
        meta_optimizer.zero_grad()

        batch_meta_loss = 0.0
        for instruction in instructions:
            # Prepare meta objective: after SFT, we want the model to output y_ref
            refusal_messages = [{"role": "user", "content": instruction}]
            refusal_inputs = prepare_instruction_inputs(
                refusal_messages,
                y_ref,
                tokenizer
            )

            # Move to device
            refusal_input_ids = refusal_inputs["input_ids"].to(DEVICE)
            refusal_attention_mask = refusal_inputs["attention_mask"].to(DEVICE)
            refusal_labels = refusal_inputs["labels"].to(DEVICE)

            # Calculate meta objective loss on the SFT'd model
            meta_obj_loss = calculate_loss(
                simulation_model,
                refusal_input_ids,
                refusal_attention_mask,
                refusal_labels
            )
            batch_meta_loss += meta_obj_loss

        # Average loss over batch
        batch_meta_loss = batch_meta_loss / len(instructions)

        # First-order MAML: Backprop through simulation_model to protected_model
        # We need to compute gradients w.r.t. protected_model's parameters
        # In first-order MAML, we ignore second-order derivatives
        batch_meta_loss.backward()

        # Map gradients from simulation_model back to protected_model
        for p_protected, p_simulation in zip(protected_model.parameters(), simulation_model.parameters()):
            if p_simulation.grad is not None:
                if p_protected.grad is None:
                    p_protected.grad = p_simulation.grad.clone()
                else:
                    p_protected.grad += p_simulation.grad

        # Meta-update
        meta_optimizer.step()

        epoch_meta_loss += batch_meta_loss.item()

        # Clean up simulation model
        del simulation_model
        torch.cuda.empty_cache()

        # Update progress bar
        progress_bar.set_postfix({
            "meta_loss": batch_meta_loss.item(),
            "avg_meta_loss": epoch_meta_loss / (batch_idx + 1)
        })

    avg_epoch_loss = epoch_meta_loss / len(train_dataloader)
    print(f"\nEpoch {epoch + 1} completed. Average meta loss: {avg_epoch_loss:.4f}")

# ======================== Save Protected Model ========================
output_dir = Path("./protected_model")
output_dir.mkdir(exist_ok=True)
protected_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nProtected model saved to {output_dir}")
