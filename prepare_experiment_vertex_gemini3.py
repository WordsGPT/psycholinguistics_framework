"""
This script is designed to prepare batches to be executed as "Batch Inferences" in Vertex AI (Google Cloud)
Batches must be uploaded to Google Cloud Buckets and executed from Vertex console
"""

import os
import sys
import jsonlines
import pandas as pd

from utils import load_config, read_txt
from datetime import datetime


def load_word_list(file_path: str, column_name: str) -> list:
    print(f"Loading word list from {file_path} column {column_name}...")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding='iso-8859-1')
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
    word_list = df[column_name].tolist()
    print(f"Successfully loaded {len(word_list)} words.")
    return word_list

def get_tasks(word_list: list,
    experiment_path: str,
    prompt: str,
    model_version: str = "gpt-4o-2024-08-06",
    temperature: int = 0,
    logprobs: bool = True,
    top_logprobs: int = 5,
    prompt_key: str = "{WORD}",
    company: str = "OpenAI",
    ft_dir: str = None) -> list:
    if company == "Google":
        return get_tasks_gemini(word_list, experiment_path, prompt, model_version, temperature, logprobs, top_logprobs,prompt_key)
    else:
        raise ValueError(f"Unknown company: {company}")

def get_tasks_gemini(
    word_list: list,
    experiment_path: str,
    prompt: str,
    model_version: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    logprobs: bool = True,
    top_logprobs: int = 5,
    prompt_key: str = "{WORD}",
) -> list:
    """
    Genera una lista de tasks con la estructura oficial de la API de Google Gemini.
    """
    tasks = []
    for counter, word in enumerate(word_list, start=1):
        task = {
            "key": f"{word}",
            "request": {
                "contents": [
                    {
                        "role": "user", 
                        "parts": [
                            {
                                "text": prompt.replace(prompt_key, str(word))
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "temperature": temperature #,
                    # "response_logprobs": logprobs,
                    # "logprobs": top_logprobs
                }
            }
        }
        tasks.append(task)
    return tasks

def create_batches(
    tasks: list, experiment_path: str, run_prefix: str, chunk_size: int = 50000
):
    os.makedirs(f"{experiment_path}/batchesA", exist_ok=True)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
    list_of_tasks = [
        tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)
    ]
    list_of_batch_names = []
    for index, tasks in enumerate(list_of_tasks):
        batch_name = f"batch_{index}_{date_string}.jsonl"
        list_of_batch_names.append(batch_name)
        with jsonlines.open(f"{experiment_path}/batchesA/{run_prefix}_{batch_name}", "w") as file:
                file.write_all(tasks)
    return list_of_batch_names


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            EXPERIMENT_NAME = sys.argv[2]
        else:
            EXPERIMENT_NAME = "original"
    else:
        print(
            "Provide as arguments the experiment path and optionally the experiment name, i.e.: python3 prepare_experiment.py <EXPERIMENT_PATH> <EXPERIMENT_NAME>."
        )
        exit()

    # prepare data
    config_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="experiments",
        name=EXPERIMENT_NAME,
    )

    word_list = load_word_list(
        file_path=f"{EXPERIMENT_PATH}/data/{config_args['dataset_path']}",
        column_name=config_args.get("dataset_column", "word"),
    )

    # prepare batch
    prompt = read_txt(
        file_path=f"{EXPERIMENT_PATH}/prompts/{config_args['prompt_path']}"
    )
    tasks = get_tasks(
        word_list=word_list,
        experiment_path=EXPERIMENT_PATH,
        prompt=prompt,
        prompt_key=f"{{{config_args['dataset_column']}}}",
        model_version=config_args["model_name"],
        company=config_args["company"],
        ft_dir=config_args.get("ft_dir", None),
        top_logprobs=config_args.get("top_logprobs", 5),
    )
    list_of_batch_names = create_batches(
        tasks=tasks,
        experiment_path=EXPERIMENT_PATH,
        run_prefix=EXPERIMENT_NAME #,
        #chunk_size = 1
    )
