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