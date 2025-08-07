"""
This script is designed to create a fine-tuning dataset for the OpenAI API. It reads a configuration file and a dataset from an Excel file, then generates a JSONL file formatted for fine-tuning. Additionally, it splits the dataset into training and testing sets.

Key Components:
- `create_division_dataset`: Splits the dataset into training and testing sets and saves them as Excel and CSV files.
- `create_jsonl`: Converts a DataFrame into a JSONL file with formatted prompts and responses.

Important Considerations:
1. Configuration File: Ensure that the `config_finetuning.yaml` file is correctly set up with the necessary parameters such as `data_path`, `prompt_question`, `prompt_answer`, `percentage`, and `random_state`.
2. Excel File: The Excel file should be located in the specified experiment directory and contain the data to be used for fine-tuning.

Usage:
Run the script from the command line with the experiment path as an argument:
    python create_finetuning_dataset.py <EXPERIMENT_PATH>

Example:
    python create_finetuning_dataset.py my_experiment

This will create a JSONL file for fine-tuning and split the dataset into training and testing sets within the specified experiment directory.
"""

import json
import os
import sys
from datetime import datetime
import re

import pandas as pd

from utils import load_config




def create_division_dataset(
    df: pd.DataFrame,
    experiment_path: str,
    percentage: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = df.sample(frac=percentage, random_state=random_state)
    test = df.drop(train.index)

    os.makedirs(f"{experiment_path}/finetuning", exist_ok=True)

    train.to_excel(
        f"{experiment_path}/finetuning/train_split_{percentage*100}_randomstate_{random_state}.xlsx",
        index=False,
    )
    test.to_excel(
        f"{experiment_path}/finetuning/test_split_{100-percentage*100}_randomstate_{random_state}.xlsx",
        index=False,
    )
    return train, test


def format_template_partial(template: str, entry: dict) -> str:
    """
    Replaces only the keys available in `entry`, leaving the missing ones intact.
    """
    def replace(match):
        key = match.group(1)
        if key in entry:
            return str(entry[key])
        else:
            return match.group(0)  # lets {key} without replacement

    return re.sub(r"{(\w+)}", replace, template)



def create_jsonl(
    df: pd.DataFrame,
    experiment_path: str,
    prompt_path: str,
    prompt_assistant_path: str,
    answer_column: str,
    dataset_name: str,
):
    # Load the prompt template from the file
    with open(f"{experiment_path}/prompts/{prompt_path}", "r") as file:
        prompt_template = file.read()
        print(prompt_template)
    if prompt_assistant_path:
        with open(f"{experiment_path}/prompts/{prompt_assistant_path}", "r") as file:
            prompt_answer_template = file.read()
            print(prompt_answer_template)
    else:
        prompt_answer_template = ""

    data = df.to_dict(orient="records")
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(
        f"{experiment_path}/finetuning/{dataset_name.split('.')[0]}_{current_time_str}.jsonl",
        "w",
    ) as file:
        for entry in data:
            entry[answer_column] = str(round(entry[answer_column], 2))
            user_content = format_template_partial(prompt_template, entry)
            if prompt_answer_template:
                assistant_content = format_template_partial(
                    prompt_answer_template, entry
                )
            else:
                assistant_content = entry[answer_column]
            json_line = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ]
            }
            # Write each dictionary as a jsonl line
            file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print("JSONL file created successfully.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            FT_NAME = sys.argv[2]
    else:
        print(
            "Provide as arguments the experiment path and optionally the fine-tuning model name, i.e.: python3 create_finetuning_dataset.py <EXPERIMENT_PATH> <FT_NAME>"
        )
        exit()

    config_finetuning_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="finetuning",
        name=FT_NAME,
    )

    full_ft_dataset_path = (
        f"{EXPERIMENT_PATH}/data/{config_finetuning_args['ft_dataset_path']}"
    )
    if full_ft_dataset_path.endswith(".csv"):
        df = pd.read_csv(full_ft_dataset_path)
    elif full_ft_dataset_path.endswith(".xlsx"):
        df = pd.read_excel(full_ft_dataset_path)
    else:
        print(f"The file {full_ft_dataset_path} is not a CSV or Excel file.")
        exit()

    train, test = create_division_dataset(
        df=df,
        experiment_path=EXPERIMENT_PATH,
        percentage=config_finetuning_args["train_split_percentage"],
        random_state=config_finetuning_args["random_state"],
    )

    create_jsonl(
        df=train,
        experiment_path=EXPERIMENT_PATH,
        prompt_path=config_finetuning_args["prompt_path"],
        prompt_assistant_path=config_finetuning_args.get("prompt_assistant_path", ""),
        answer_column=config_finetuning_args["answer_column"],
        dataset_name=config_finetuning_args["dataset_finetune_name"],
    )
