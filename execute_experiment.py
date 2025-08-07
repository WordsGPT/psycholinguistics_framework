"""
This script is designed to execute tasks for psycholinguistic experiments using the OpenAI API. It reads a list of batch files, submits them to the OpenAI API for processing, and retrieves the results once the processing is complete.

Key Components:
- `execute_tasks`: Submits batch files to the OpenAI API and returns a list of job IDs.
- `retrieve_batch_job`: Checks the status of batch jobs to determine if they are complete.
- `save_results`: Saves the results of completed batch jobs to JSONL files.
- `read_list_of_batchs`: Reads the list of batch files from the specified directory.

Important Considerations:
1. Configuration File: Ensure that the `config_experiment.yaml` file is correctly set up with the necessary parameters.
2. Environment Variables: The `apis.env` file must contain valid OpenAI API credentials.
3. Batch Files: Ensure that the batch files are correctly prepared and located in the specified experiment directory.

Usage:
Run the script from the command line with the experiment name as an argument:
    python execute_experiment.py <EXPERIMENT_PATH>

Example:
    python execute_experiment.py my_experiment

This will execute the tasks in the batch files and save the results in the `results` directory within the specified experiment directory.
"""

import os
import sys
import time
from datetime import datetime

from utils import load_config, openai_login


def read_list_of_batches(folder_name: str, run_prefix: str) -> list:
    file_names = [
        file
        for file in os.listdir(f"./{folder_name}/batches")
        if file.startswith(run_prefix)
    ]
    print(f"Batches to run: {file_names}")
    return file_names


def execute_tasks(list_of_batch_names: list, experiment_path: str) -> list:
    list_of_job_ids = []
    for file_name in list_of_batch_names:
        batch_file = client.files.create(
            file=open(f"{experiment_path}/batches/{file_name}", "rb"), purpose="batch"
        )
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"# File Named: {file_name} has been submitted for processing")
        print(f"# Batch Job ID: {batch_job.id}")
        print()
        list_of_job_ids.append(batch_job.id)
    return list_of_job_ids


def retrieve_batch_job(batch_job_id):
    finised = True
    for batch_job_id in list_of_job_ids:
        if client.batches.retrieve(batch_job_id).status != "completed":
            print(
                f"Batch Job ID: {batch_job_id} is still in progress. Status: {client.batches.retrieve(batch_job_id).status}"
            )
            finised = False
            break
    return finised


def save_results(list_of_job_ids: list, experiment_path: str, run_prefix: str):
    os.makedirs(f"{experiment_path}/results", exist_ok=True)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")

    for index, batch_job_id in enumerate(list_of_job_ids):
        batch_results_id = client.batches.retrieve(batch_job_id).output_file_id
        result = client.files.content(batch_results_id).content
        with open(
            f"{experiment_path}/results/{run_prefix}_results_{index}_{date_string}.jsonl",
            "wb",
        ) as file:
            file.write(result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            EXPERIMENT_NAME = sys.argv[2]
        else:
            EXPERIMENT_NAME = "original"
    else:
        print(
            "Provide as arguments the experiment path and optionally the experiment name, i.e.: python3 execute_experiment.py <EXPERIMENT_PATH> <EXPERIMENT_NAME>."
        )
        exit()

    # login
    client = openai_login()

    # prepare data
    config_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="experiments",
        name=EXPERIMENT_NAME,
    )

    # load list of batches and execute tasks
    list_of_batch_names = read_list_of_batches(
        folder_name=EXPERIMENT_PATH, run_prefix=EXPERIMENT_NAME
    )
    list_of_job_ids = execute_tasks(
        list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH
    )

    # wait for batch jobs to finish
    while not retrieve_batch_job(batch_job_id=list_of_job_ids):
        wait_time = 300
        print(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
        pass

    save_results(
        list_of_job_ids=list_of_job_ids,
        experiment_path=EXPERIMENT_PATH,
        run_prefix=EXPERIMENT_NAME,
    )
