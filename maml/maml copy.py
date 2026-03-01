import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

harmful_data_path = Path("./for_finetuning/harmful_5394.json")
harmful_dataset = json.loads(harmful_data_path.read_text())
harmful_dataset = Dataset.from_list(harmful_dataset)
y_ref = "I'm sorry but i cannot provide any information about your request."

model_id = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id)

protected_model = base_model.copy()

for harmful_sample in harmful_dataset:
    instruction = harmful_sample["instruction"]
    y_harm = harmful_sample["output"]
    
    simulation_model = protected_model.copy()

    harmful_messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": y_harm},
    ]

    harmful_inputs = tokenizer.apply_chat_template(
        harmful_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(simulation_model.device)

    sft_loss = calculate_loss(simulation_model, harmful_inputs)

    sft_loss.backward()
    simulation_model.update()

    meta_obj_loss = calculate_loss(simulation_model, harmful_inputs, y_ref)

    meta_obj_loss.backward()
    protected_model.update(simulation_model.gradient)

    

