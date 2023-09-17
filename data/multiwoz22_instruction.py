from datasets import load_dataset
import json

def process_data(example):
    turns = example["turns"][0]
    ret = {
        "dialogue_id": [],
        "turn_id": [],
        "services": [],
        "context": [],
        "domain": [],
        "slot_name": [],
        "slot_value": [],
    }

    current_context = []
    for i in range(len(turns["utterance"])):
        utter = turns["utterance"][i]
        speaker = turns["speaker"][i]
        turn_id = turns["turn_id"][i]
        frame = turns["frames"][i]

        active_slots = {}
        for state_per_domain in frame["state"]:
            for slot_name, slot_value in zip(state_per_domain["slots_values"]["slots_values_name"], state_per_domain["slots_values"]["slots_values_list"]):
                active_slots[slot_name] = slot_value

        if speaker == 0: # USER
            current_context.append("[USER]: "+utter)
        
            for domain in frame["service"]:
                for slot in domain2slot[domain]:
                    # Fixed for all slots
                    ret["dialogue_id"].append(example["dialogue_id"])
                    ret["turn_id"].append(turn_id)
                    ret["services"].append(example["services"])
                    ret["context"].append("\n".join(current_context))

                    ret["domain"].append(domain)

                    ret["slot_name"].append(slot["name"])
                    ret["slot_value"].append(["NONE"] if slot["name"] not in active_slots.keys() else active_slots[slot["name"]])
                    
        else:
            current_context.append("[ASSISTANT]: "+utter)
    return ret

def convert_to_instruction_following_prompts(examples):
    
    template_non_categorical = """Based on the input dialogue between the user and the assistant, answer \"{slot_desc}\". If it\'s not mentioned in the dialogue, please answer NONE. """
    template_categorical = """Based on the input dialogue between the user and the assistant, choose the correct answer for \"{slot_desc}\" from {slot_space}. If it\'s not mentioned in the dialogue, please choose NONE. """
    instructions = []
    inputs = []
    outputs = []
    for i in range(len(examples["slot_name"])):
        for slot_val in examples["slot_value"][i]:
            slot_name = examples["slot_name"][i]
            slot_value = slot_val
            slot_desc = slot2desc[slot_name]
            slot_space = slot2space.get(slot_name, None)
            if slot_space is not None:
                slot_space = "[" + ", ".join(slot_space+["NONE"]) + "]"

            instructions.append(template_categorical.format(slot_desc=slot_desc, slot_space=slot_space) if slot_space is not None else template_non_categorical.format(slot_desc=slot_desc))
            inputs.append(examples["context"][i])
            outputs.append(slot_value)

    return {
        "instruction": instructions,
        "input": inputs,
        "output": outputs,
    }


if __name__ == "__main__":
    ds = load_dataset("multi_woz_v22")

    with open("./schema.json", "r") as f:
        schema = json.load(f)

    domain2slot = {}
    slot2desc = {}
    slot2space = {}
    for domain_json in schema:
        domain2slot[domain_json["service_name"]] = domain_json["slots"]
        for slot in domain_json["slots"]:
            slot2desc[slot["name"]] = slot["description"]
            slot2space[slot["name"]] = slot["possible_values"] if slot["is_categorical"] else None

    processed_ds = ds.map(process_data, batched=True, batch_size=1, remove_columns=ds["train"].column_names, num_proc=8)
    instruction_ds = processed_ds.map(convert_to_instruction_following_prompts, batched=True, batch_size=100, remove_columns=processed_ds["train"].column_names, num_proc=8)

    print(instruction_ds)
    print("\n\n")
    print("Example datapoint:\n")
    print(instruction_ds["train"][0])

    instruction_ds["train"].to_json("./MultiWOZ_2.2_instruction/train.jsonl", orient="records", lines=True)
    instruction_ds["validation"].to_json("./MultiWOZ_2.2_instruction/val.jsonl", orient="records", lines=True)
    instruction_ds["test"].to_json("./MultiWOZ_2.2_instruction/test.jsonl", orient="records", lines=True)

    processed_ds["train"].to_json("./MultiWOZ_2.2_raw/train.jsonl", orient="records", lines=True)
    processed_ds["validation"].to_json("./MultiWOZ_2.2_raw/val.jsonl", orient="records", lines=True)
    processed_ds["test"].to_json("./MultiWOZ_2.2_raw/test.jsonl", orient="records", lines=True)