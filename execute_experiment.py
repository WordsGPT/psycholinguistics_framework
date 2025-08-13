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
from google import genai
from google.genai import types

from utils import load_config, openai_login, google_login, huggingface_login

import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
logging.set_verbosity_error()
import transformers
import torch
import json
from tqdm import tqdm

def read_list_of_batches(folder_name: str, run_prefix: str) -> list:
    file_names = [
        file
        for file in os.listdir(f"./{folder_name}/batches")
        if file.startswith(run_prefix)
    ]
    print(f"Batches to run: {file_names}")
    return file_names


## OpenAI ##

def execute_tasks_openai(list_of_batch_names: list, experiment_path: str) -> list:
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


def retrieve_batch_job_openai(batch_job_ids):
    finished = True
    for batch_job_id in batch_job_ids:
        if client.batches.retrieve(batch_job_id).status != "completed":
            print(
                f"Batch Job ID: {batch_job_id} is still in progress. Status: {client.batches.retrieve(batch_job_id).status}"
            )
            finished = False
            break
    return finished

def save_results_openai(list_of_job_ids: list, experiment_path: str, run_prefix: str):
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

## Google ##
def get_batch_names_google():
    batch_jobs = client.batches.list(config={
        "page_size": 100,
    })
    for batch_job in batch_jobs:
        print(batch_job)
        print(f"Batch job: {batch_job.name}, state {batch_job.state}, created at {batch_job.create_time}")

def cancel_batch_job_google(batch_job_name: str):
    client.batches.cancel(name=batch_job_name)


def execute_tasks_google(list_of_batch_names: list, experiment_path: str) -> list:
    list_of_job_names = []
    for file_name in list_of_batch_names:
        uploaded_file = client.files.upload(
            file=open(f"{experiment_path}/batches/{file_name}", "rb"),
            config=types.UploadFileConfig(display_name=file_name, mime_type='jsonl')
        )
        print(f"Uploaded file: {uploaded_file.name}")
        batch_job = client.batches.create(
            model=config_args.get("model_name"),
            src=uploaded_file.name,
            config={
                'display_name': f"{experiment_path}_{file_name}",
            },
        )
        print(f"Created batch job: {batch_job.name}")
        list_of_job_names.append(batch_job.name)
    return list_of_job_names


def format_results_huggingface(output, tokenizer, counter, experiment_path, logprobs=5):
    gen_text = output[0]["generated_text"]
    scores = output[0]["scores"]

    char_pos = gen_text.find('{')
    if char_pos != -1:
        prefix_tokens = tokenizer.encode(gen_text[:char_pos], add_special_tokens=False)
        start_idx = len(prefix_tokens)
    else:
        start_idx=-1
    
    char_pos = gen_text.find('}')
    if char_pos != -1:
        prefix_tokens = tokenizer.encode(gen_text[:char_pos], add_special_tokens=False)
        end_idx = len(prefix_tokens)
    else:
        end_idx=-1

    if start_idx == -1:
        token_entry = []
    else:
        token_entry_list=[]
        for step_logits_list in scores[start_idx:end_idx]:
            step_logits = torch.tensor(step_logits_list)
            step_probs = torch.softmax(step_logits, dim=0)
            step_logprobs = torch.log(step_probs)
            top_probs, top_indices = torch.topk(step_probs, logprobs)
            top_logprobs = torch.log(top_probs)
            
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
            
            cur_token = top_tokens[0]
            cur_logprob = top_logprobs.tolist()[0]
            
            top_logprobs_list = []
            for t, lp in zip(top_tokens, top_logprobs.tolist()):
                top_logprobs_list.append({
                    "token": t,
                    "logprob": lp,
                    "bytes": list(t.encode("utf-8"))
                })
            token_entry = {
                "token": cur_token,
                "logprob": cur_logprob,
                "bytes": list(cur_token.encode("utf-8")),
                "top_logprobs": top_logprobs_list
            }
            token_entry_list.append(token_entry)
    json_line = {
        "id": f"{experiment_path}_task_{counter}",
        "response": {
            "status_code": -1,
            "body": {
                "model": config_args['model_name'],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": gen_text,
                            "refusal": None,
                            "annotations": []
                        },
                        "logprobs": {
                            "content": token_entry_list
                        }
                    }
                ]
            }
        }
    }
    return json_line


def execute_tasks_save_huggingface(list_of_batch_names: list, experiment_path: str, run_prefix: str, batch_size = 5) -> list:
    list_of_job_names = []
    for index, file_name in enumerate(list_of_batch_names):
        jsonl_file_path = f"{experiment_path}/batches/{file_name}"
        with jsonlines.open(jsonl_file_path, "r") as reader:
            for obj in reader:
                model_name = obj.get("model")
                temperature = obj.get("temperature")
                response_logprobs = obj.get("response_logprobs")
                logprobs = obj.get("logprobs")
                break
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
        os.makedirs(f"{experiment_path}/results", exist_ok=True)
        output_file = f"{experiment_path}/results/{run_prefix}_results_{index+1}_{date_string}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f_out:  
            with jsonlines.open(jsonl_file_path, "r") as reader:
                batch_messages = []
                counter = 0
                for obj in tqdm(reader, desc=f"Processing {jsonl_file_path}"):
                    prompt = obj.get("prompt")

                    batch_messages.append([{"role": "user", "content": prompt}])
                    if len(batch_messages) == batch_size:
                        outputs = pipeline(
                            batch_messages,
                            max_new_tokens=500,
                            temperature = temperature,
                            do_sample=False,
                            return_full_text=False,
                            return_dict_in_generate=True,
                            output_scores=response_logprobs
                        )
                        for output in outputs:
                            counter += 1
                            json_line = format_results_huggingface(output, tokenizer, counter, experiment_path, logprobs)
                            f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                            f_out.flush()
                        batch_messages=[]


def retrieve_batch_job_google(batch_job_names):
    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])
    finished = True
    for batch_job_name in batch_job_names:
        batch_job = client.batches.get(name=batch_job_name)
        print(f"Current state of {batch_job_name}: {batch_job.state.name}")
        if batch_job.state.name not in completed_states:
            finished = False
            break
    return finished

def save_results_google(list_of_job_names: list, experiment_path: str, run_prefix: str):
    os.makedirs(f"{experiment_path}/results", exist_ok=True)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    for index, batch_job_name in enumerate(list_of_job_names):
        batch_job = client.batches.get(name=batch_job_name)
        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                result_file_name = batch_job.dest.file_name
                print(f"Results are in file: {result_file_name}")

                print("Downloading result file content...")
                file_content = client.files.download(file=result_file_name)
                save_path = f"{experiment_path}/results/{run_prefix}_results_{index}_{date_string}.jsonl"
                with open(save_path, "wb") as f:
                    f.write(file_content)
        else:
            print(f"Batch job {batch_job_name} did not succeed.")

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


    # prepare data
    config_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="experiments",
        name=EXPERIMENT_NAME,
    )
    print(config_args)

    # load list of batches and execute tasks
    list_of_batch_names = read_list_of_batches(
        folder_name=EXPERIMENT_PATH, run_prefix=EXPERIMENT_NAME
    )

    company = config_args.get("company")
    if company == "OpenAI":
        # login
        client = openai_login()
        list_of_job_ids = execute_tasks_openai(
            list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH
        )

        # wait for batch jobs to finish
        while not retrieve_batch_job_openai(batch_job_ids=list_of_job_ids):
            wait_time = 300
            print(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass

        save_results_openai(
            list_of_job_ids=list_of_job_ids,
            experiment_path=EXPERIMENT_PATH,
            run_prefix=EXPERIMENT_NAME,
        )
    elif company == "Google":
        client = google_login()
        list_of_job_ids = execute_tasks_google(
            list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH
        )
        # wait for batch jobs to finish
        while not retrieve_batch_job_google(batch_job_names=list_of_job_ids):
            wait_time = 300
            print(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass

        save_results_google(
            list_of_job_names=list_of_job_ids,
            experiment_path=EXPERIMENT_PATH,
            run_prefix=EXPERIMENT_NAME,
        )
    elif company == "HuggingFace":
        huggingface_login()
        execute_tasks_save_huggingface(
            list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH, run_prefix=EXPERIMENT_NAME
        )
    else:
        print(f"Company {company} is not supported.")
