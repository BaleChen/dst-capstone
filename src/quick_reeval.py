import json
import pandas as pd
import argparse
from copy import deepcopy

def transform_test_data_to_single_dict(test_data):
    ret_test_data = {}
    for dialogue in test_data:
        dialogue_id = dialogue['dialogue_idx'].split('.')[0]
        for turn in dialogue['dialogue']:
            turn_id = turn['turn_idx'] * 2
            ret_test_data[f'{dialogue_id}-{turn_id}'] = turn
    return ret_test_data

def compute_metrics(results, num_all_none_turns=0, num_new_fn_turns=0):
    # loop through all the dictionary items
    total, joint_acc, F1_pred = 0, 0, 0
    precision, recall = 0, 0
    for dialogue_turn_id, result in results.items():
        pred_state, true_state = result["pred_state"], result["true_state"]
        total += 1

        turn_correct, turn_f1, (turn_precision, turn_recall), jga_flag = compute_turn_acc_and_f1(pred_state, true_state)

        if jga_flag:
            joint_acc += 1
        F1_pred += turn_f1
        precision += turn_precision
        recall += turn_recall
    
    precision = precision / (total + num_new_fn_turns)
    recall = recall / (total + num_new_fn_turns)
    joint_acc = (joint_acc + num_all_none_turns) / (total + num_all_none_turns) # NOTE: Add back the all none turns
    F1_score = F1_pred / (total + num_new_fn_turns)
    return {"joint_acc": joint_acc, "slot_f1": F1_score, "precision": precision, "recall": recall} 

def compute_turn_acc_and_f1(pred_state, true_state):
    """Compute the turn-level accuracy, precision, recall, and F1 score."""
    turn_correct, tp, fp, fn = 0, 0, 0, 0
    for slot, pred_value in pred_state.items():
        true_values = true_state[slot]
        if pred_value in true_values:
            turn_correct += 1
            tp += 1
        elif pred_value == "None":
            fn += 1
        elif true_values == ["None"]:
            fp += 1
    turn_precision = tp / (tp + fp) if tp + fp > 0 else 0
    turn_recall = tp / (tp + fn) if tp + fn > 0 else 0
    turn_f1 = 2 * turn_precision * turn_recall / (turn_precision + turn_recall) if turn_precision + turn_recall > 0 else 0
    
    return turn_correct, turn_f1, (turn_precision, turn_recall), fp+fn == 0

def main(args):
    with open(args.pred_file) as f:
        preds = json.load(f)
    with open(args.new_test_data) as f:
        new_test_data = json.load(f)
        new_test_data = transform_test_data_to_single_dict(new_test_data)
    old_test_data = pd.read_json(args.old_test_data, lines=True)
    old_test_data_turn_set = set(old_test_data["dialogue_turn_id"])
    pred_turn_set = set(preds.keys())
    num_all_none_turns = len(old_test_data_turn_set) - len(pred_turn_set) # this is a fix to our previous evaluation

    # find if the slots in (old_test - pred) has been modified in new_test
    modified_wrong_cnt = 0 # how many slot values have been modified to be not none and cause our predictions to be wrong
    modified_fn = 0 # how many slot values have been modified to be not none (which would be a fn count)
    for turn in (old_test_data_turn_set - pred_turn_set):
        try:
            new_test_turn_label = new_test_data[turn]['turn_label']
        except KeyError:
            continue
        temp_fn = len(new_test_turn_label)
        if temp_fn > 0:
            modified_wrong_cnt += 1
        modified_fn += temp_fn

    # update true states
    new_preds = deepcopy(preds)
    for dialogue_turn_id, values in preds.items():
        if dialogue_turn_id not in new_test_data:
            continue
        old_belief_state = values["true_state"]
        new_belief_state = new_test_data[dialogue_turn_id]['belief_state']
        domain = list(values["true_state"].keys())[0].split("-")[0]

        for slot_value_list in new_belief_state:
            slot = slot_value_list[0]
            if " " in slot: # fix data format inconsistency
                slot = slot.replace(" ", "")
            new_value = slot_value_list[1]
            if domain not in slot: # find only the requested domain
                continue

            if new_value == "" or new_value == "not mentioned":
                new_value = "None"
                if slot in old_belief_state and old_belief_state[slot] != ["None"]:
                    # print(f"Changing slot value {new_preds[dialogue_turn_id]['true_state'][slot]} to None")
                    new_preds[dialogue_turn_id]["true_state"][slot] = ["None"]
            else:
                # print(f"Changing slot value for {slot} to {new_value}")
                new_preds[dialogue_turn_id]["true_state"][slot] = [new_value]
                if slot not in new_preds[dialogue_turn_id]["pred_state"]:
                    new_preds[dialogue_turn_id]["pred_state"][slot] = "None"
            # NOTE
            # update the slot values
            # Old        New
            # DNE        "None" -> do nothing
            # DNE        "something" -> update to ["something"]
            # ["None"]   "None" -> do nothing
            # ["None"]   "something" -> update to ["something"]
            # ["something"] "None" -> update to ["None"]
            # ["something"] "something_else" -> update to ["something_else"]
            # ["something"] "something" -> update to ["something"]

    metrics_24 = compute_metrics(new_preds, num_all_none_turns, modified_wrong_cnt)
    metrics_22 = compute_metrics(preds, num_all_none_turns)
    print(f"Metrics for MWZ_24: {metrics_24}")
    print(f"Metrics for MWZ_22: {metrics_22}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default=None, required=True)
    parser.add_argument('--new_test_data', type=str, default=None, required=True)
    parser.add_argument('--old_test_data', type=str, default=None, required=True)
    args = parser.parse_args()
    main(args)