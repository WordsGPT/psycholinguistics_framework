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

from utils import load_config, openai_login, google_login, huggingface_login, vertec_login

import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
logging.set_verbosity_error()
import transformers
import torch
import json
from tqdm import tqdm
from peft import PeftModel
import vertexai
from google.cloud import aiplatform, storage
from vertexai.tuning import sft


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


def add_role_to_jsonl(input_file: str, output_file: str, default_role: str = "user"):
    """
    Adjusts the JSONL data format to meet the requirements for batch processing using a fine-tuned Google model.
    """
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)
            if "request" in data and "contents" in data["request"]:
                for content in data["request"]["contents"]:
                    if "role" not in content:
                        content["role"] = default_role
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")


def execute_tasks_google(list_of_batch_names: list, experiment_path: str) -> list:
    if config_args.get("model_name").startswith('projects'):
        resource_names = []
        for file_name in list_of_batch_names:
            fine_tuning_job_id = config_args.get("model_name")
            # projects/139736761406/locations/us-central1/tuningJobs/5827228369548214272
            parts = fine_tuning_job_id.split('/')
            PROJECT_ID = parts[1]
            LOCATION = parts[3]
            tuning_job_id = parts[5]
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            BUCKET_NAME = config_args.get("bucket_name")
            date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
            
            GCS_OUTPUT_PREFIX = f"gs://{BUCKET_NAME}/batch_outputs_{date_string}/"
            LOCAL_FILE_original = f"{experiment_path}/batches/{file_name}"
            LOCAL_FILE = f"{experiment_path}/batches/Add_role_{file_name}"
            add_role_to_jsonl(LOCAL_FILE_original, LOCAL_FILE)
            
            sft_job = sft.SupervisedTuningJob(fine_tuning_job_id)
            client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=LOCATION
            )
            
            tj = client.tunings.get(name=fine_tuning_job_id)
            MODEL_ID = tj.tuned_model.model.split('/')[-1]
            
    
            def upload_to_gcs(local_file, bucket_name, destination_blob_name):
                client = storage.Client(project=PROJECT_ID)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(local_file)
                print(f"Uploaded file: gs://{bucket_name}/{destination_blob_name}")
                
            upload_file_path = f"batch_inputs/{file_name.replace('.jsonl','')}_{date_string}.jsonl"
            upload_to_gcs(LOCAL_FILE, BUCKET_NAME, upload_file_path)
            GCS_INPUT_PATH = f"gs://{BUCKET_NAME}/batch_inputs/{file_name.replace('.jsonl','')}_{date_string}.jsonl"
            
            aiplatform.init(project=PROJECT_ID, location=LOCATION)
            
            model_resource_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}"
            
            batch_prediction_job = aiplatform.BatchPredictionJob.create(
                job_display_name="gemini-finetune-batch",
                model_name=model_resource_name,
                gcs_source=[GCS_INPUT_PATH],
                gcs_destination_prefix=GCS_OUTPUT_PREFIX,
                instances_format="jsonl",
                predictions_format="jsonl"
            )
            resource_names.append(batch_prediction_job.resource_name)
        return resource_names
    
    else:
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
    token_entry_list=[]
    for step_logits_list in scores:
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





def execute_tasks_save_model_hf_or_local(list_of_batch_names: list, experiment_path: str, run_prefix: str, batch_size=5) -> list:
    list_of_job_names = []

    for index, file_name in enumerate(list_of_batch_names):
        jsonl_file_path = f"{experiment_path}/batches/{file_name}"

        # Read configuration from the first entry of the JSONL file
        with jsonlines.open(jsonl_file_path, "r") as reader:
            for obj in reader:
                model_name = obj.get("model")
                ft_dir = obj.get("ft_dir", None)  # optional LoRA directory
                temperature = obj.get("temperature")
                response_logprobs = obj.get("response_logprobs")
                logprobs = obj.get("logprobs")
                break

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Apply LoRA if directory is provided
        if ft_dir:
            ft_dir_final = f"{experiment_path}/finetuning/cache_model/{ft_dir}"
            if not os.path.isdir(ft_dir_final):
                ft_dir_final = f"{experiment_path}/finetuning_final/{ft_dir}/cache_model/{ft_dir}"
            if not os.path.isdir(ft_dir_final):
                raise ValueError(f"LoRA directory {ft_dir_final} does not exist.")
            print(f"Loading LoRA-finetuned model from {ft_dir_final}")
            model = PeftModel.from_pretrained(base_model, ft_dir_final)
            tokenizer = AutoTokenizer.from_pretrained(ft_dir_final)
        else:
            model = base_model
            print(f"Using base model {model_name} (no LoRA applied)")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create text-generation pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

        # Prepare output file
        date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
        os.makedirs(f"{experiment_path}/results", exist_ok=True)
        output_file = f"{experiment_path}/results/{run_prefix}_results_{index+1}_{date_string}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f_out:
            with jsonlines.open(jsonl_file_path, "r") as reader:
                batch_messages = []
                counter = 0

                # Process prompts in batches
                for obj in tqdm(reader, desc=f"Processing {jsonl_file_path}"):
                    prompt = obj.get("prompt")
                    batch_messages.append([{"role": "user", "content": prompt}])

                    if len(batch_messages) == batch_size:
                        outputs = pipeline(
                            batch_messages,
                            max_new_tokens=500,
                            temperature=temperature,
                            do_sample=True if temperature > 0 else False,
                            return_full_text=False,
                            return_dict_in_generate=True,
                            output_scores=response_logprobs
                        )
                        for output in outputs:
                            counter += 1
                            json_line = format_results_huggingface(output, tokenizer, counter, experiment_path, logprobs)
                            f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                            f_out.flush()
                        batch_messages = []

                # Process any remaining messages
                if len(batch_messages) > 0:
                    outputs = pipeline(
                        batch_messages,
                        max_new_tokens=500,
                        temperature=temperature,
                        do_sample=True if temperature > 0 else False,
                        return_full_text=False,
                        return_dict_in_generate=True,
                        output_scores=response_logprobs
                    )
                    for output in outputs:
                        counter += 1
                        json_line = format_results_huggingface(output, tokenizer, counter, experiment_path, logprobs)
                        f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                        f_out.flush()
                    batch_messages = []


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

def retrieve_batch_job_google_finetuned(batch_job_names):

    finished_states = {
        aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED,
        aiplatform.gapic.JobState.JOB_STATE_FAILED,
        aiplatform.gapic.JobState.JOB_STATE_CANCELLED,
        aiplatform.gapic.JobState.JOB_STATE_EXPIRED
    }
    parts = batch_job_names[0].split('/')
    PROJECT_ID = parts[1]
    LOCATION = parts[3]
    finished=True
    client = aiplatform.gapic.JobServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    )
    jobs = []
    for batch_job_name in batch_job_names:
        parts = batch_job_names[0].split('/')
        BATCH_JOB_ID = parts[5]
        name = client.batch_prediction_job_path(
            project=PROJECT_ID, location=LOCATION, batch_prediction_job=BATCH_JOB_ID
        )
        job = client.get_batch_prediction_job(name=name)
        jobs.append(job)
        if job.state not in finished_states:
            finished = False
            break
    return finished, jobs


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


def save_results_google_finetuned(jobs: list, experiment_path: str, run_prefix: str):
    os.makedirs(f"{experiment_path}/results", exist_ok=True)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")

    
    for index, job in enumerate(jobs):
        
        if job.state.name == 'JOB_STATE_SUCCEEDED':
            gcs_output_dir = job.output_info.gcs_output_directory
            bucket_name = gcs_output_dir.replace("gs://", "").split("/")[0]
            prefix = "/".join(gcs_output_dir.replace("gs://", "").split("/")[1:])
            
            storage_client = storage.Client(project=config_args['project_name'])
            bucket = storage_client.bucket(bucket_name)
            
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith("predictions.jsonl"):
                    save_path = f"{experiment_path}/results/{run_prefix}_results_{index}_{date_string}.jsonl"
                    print("Downloading result file content...")
                    blob.download_to_filename(save_path)

        else:
            print(f"Batch job {job.name} did not succeed.")


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
        if config_args.get("model_name").startswith('projects'):
            vertec_login()
            list_of_job_ids = execute_tasks_google(
                list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH
            )
            while True:
                finished, jobs = retrieve_batch_job_google_finetuned(batch_job_names=list_of_job_ids)
                if finished:
                    break
                wait_time = 300
                print(f"Waiting for {wait_time} seconds")
                time.sleep(wait_time)
                
            save_results_google_finetuned(
                jobs=jobs,
                experiment_path=EXPERIMENT_PATH,
                run_prefix=EXPERIMENT_NAME,
            )

        else:
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
        execute_tasks_save_model_hf_or_local(
            list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH, run_prefix=EXPERIMENT_NAME
        )
    elif company == "Local":
        execute_tasks_save_model_hf_or_local(
            list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH, run_prefix=EXPERIMENT_NAME
        )
    else:
        print(f"Company {company} is not supported.")
