"""
This script is designed to execute a fine-tuning job using the OpenAI API. It reads a configuration file to obtain the necessary parameters and submits a fine-tuning job with a specified dataset.

Key Components:
- `create_finetune`: Submits a fine-tuning job to the OpenAI API using the specified dataset and model, you can check the status of the job at https://platform.openai.com/finetune/ftjob-QPje3bZiOQ4shvQs6rKnRkyn.

Important Considerations:
1. Configuration File: Ensure that the `config_finetuning.yaml` file is correctly set up with the necessary parameters such as `dataset_finetune_name`, `model_name`, and `especial_suffix`.
2. Environment Variables: The `apis.env` file must contain valid OpenAI API credentials.
3. Dataset File: The dataset file should be located in the specified experiment directory and formatted correctly for fine-tuning.

Usage:
Run the script from the command line with the experiment path as an argument:
    python execute_finetune.py <EXPERIMENT_PATH>

Example:
    python execute_finetune.py my_experiment

This will submit a fine-tuning job using the specified dataset and model configuration.
"""

import sys

from utils import load_config, openai_login, read_yaml


def create_finetune(file_path: str, model_name: str, suffix: str):
    file_object = client.files.create(file=open(file_path, "rb"), purpose="fine-tune")
    finetune_job = client.fine_tuning.jobs.create(
        training_file=file_object.id, model=model_name, suffix=suffix
    )
    return finetune_job


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            FT_NAME = sys.argv[2]
    else:
        print(
            "Provide as arguments the experiment path and the fine-tuned model name, i.e.: python3 execute_finetune.py <EXPERIMENT_PATH> <FT_NAME>."
        )
        exit()

    config_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="finetuning",
        name=FT_NAME,
    )

    client = openai_login()

    finetune_job = create_finetune(
        file_path=f"{EXPERIMENT_PATH}/finetuning/{config_args['dataset_finetune_name']}",
        model_name=config_args["model_name"],
        suffix=config_args["especial_suffix"],
    )

    # https://platform.openai.com/finetune/ftjob-QPje3bZiOQ4shvQs6rKnRkyn
