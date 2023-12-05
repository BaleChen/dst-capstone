from time import gmtime, strftime

def prepare_exp_name(
    model_args,
    training_args,
    data_args,
    lora_args,
):
    exp_name = strftime("%Y-%m-%d-%H:%M", gmtime())
    exp_name += f"_{model_args.model_name_or_path.split('/')[-1]}"
    exp_name += f"_pdbs{training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps}"
    exp_name += f"_lr{training_args.learning_rate}"

    # TODO specify the dataset format
    return exp_name    

def first_difference_index(list1, list2):
    # Find the minimum length of the two lists
    min_length = min(len(list1), len(list2))

    # Iterate through the lists up to the length of the shorter list
    for i in range(min_length):
        if list1[i] != list2[i]:
            return i  # Return the index of the first differing element

    # If lists are of the same length and no differences are found, return -1
    return -1