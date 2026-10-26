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
import re
import gc
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
from transformers import AutoProcessor



def read_list_of_batches(folder_name: str, run_prefix: str) -> list:
    file_names = [
        file
        for file in os.listdir(f"./{folder_name}/batches")
        # Require "_batch_" right after the name, so "<name>_no_logs_batch_..."
        # is not picked up when running "<name>"
        if file.startswith(f"{run_prefix}_batch_")
    ]
    # Numeric order (os.listdir order is arbitrary and "batch_10" < "batch_2" as text)
    file_names.sort(key=lambda name: int(re.search(r"_batch_(\d+)_", name).group(1)))
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


def execute_tasks_google(client, list_of_batch_names: list, experiment_path: str) -> list:
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


def logprob_entries_from_scores(step_scores, tokenizer, top_k):
    """
    Build the OpenAI-style logprobs list for one generated sequence.
    step_scores: tensor [n_steps, vocab] with the scores of each generated step
    (kept on the GPU, so softmax and top-k are fast).
    """
    if step_scores is None or step_scores.shape[0] == 0:
        return []
    step_logprobs = torch.log_softmax(step_scores.float(), dim=-1)
    top_logprobs, top_indices = torch.topk(step_logprobs, top_k, dim=-1)
    top_logprobs = top_logprobs.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    token_entry_list = []
    for step_lps, step_ids in zip(top_logprobs, top_indices):
        top_tokens = tokenizer.convert_ids_to_tokens(step_ids)
        top_logprobs_list = [
            {"token": t, "logprob": lp, "bytes": list(t.encode("utf-8"))}
            for t, lp in zip(top_tokens, step_lps)
        ]
        token_entry_list.append({
            "token": top_tokens[0],
            "logprob": step_lps[0],
            "bytes": list(top_tokens[0].encode("utf-8")),
            "top_logprobs": top_logprobs_list,
        })
    return token_entry_list


def format_results_huggingface(gen_text, token_entry_list, custom_id):
    json_line = {
        "id": custom_id,
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



def ensure_chat_template(tokenizer, model_name: str) -> bool:
    """
    Try to give the tokenizer a chat template.
    Tokenizers saved next to a LoRA adapter may not carry one, so we borrow it
    from the base model's tokenizer, then from its processor.
    Returns True if a chat template is available, False otherwise.
    Base (pretrained, non "-it") models such as google/gemma-4-31B have no chat
    template at all; in that case prompts are sent as plain text.
    """
    if tokenizer.chat_template:
        return True

    try:
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if base_tokenizer.chat_template:
            tokenizer.chat_template = base_tokenizer.chat_template
            print(f"Chat template copied from base tokenizer {model_name}")
            return True
    except Exception as e:
        print(f"Could not load base tokenizer for chat template: {e}")

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        if getattr(processor, "chat_template", None):
            tokenizer.chat_template = processor.chat_template
            print(f"Chat template copied from processor {model_name}")
            return True
    except Exception as e:
        print(f"Could not load processor for chat template: {e}")

    print(f"No chat template for {model_name} (base model); sending prompts as plain text.")
    return False


def parse_batch_line(obj: dict) -> dict:
    """
    Read one batch line, in either layout:
    - OpenAI style (prepare_experiment.py): {"custom_id", "body": {"model", "temperature",
      "logprobs": bool, "top_logprobs": int, "messages": [...]}}
    - flat style: {"prompt", "model", "temperature", "response_logprobs": bool, "logprobs": int, "ft_dir"}
    """
    if "body" in obj:
        body = obj["body"]
        return {
            "custom_id": obj.get("custom_id"),
            "model": body.get("model"),
            "ft_dir": obj.get("ft_dir", body.get("ft_dir")),
            "temperature": body.get("temperature"),
            "response_logprobs": body.get("logprobs"),
            "logprobs": body.get("top_logprobs"),
            "prompt": body["messages"][-1]["content"],
        }
    return {
        "custom_id": obj.get("custom_id"),
        "model": obj.get("model"),
        "ft_dir": obj.get("ft_dir"),
        "temperature": obj.get("temperature"),
        "response_logprobs": obj.get("response_logprobs"),
        "logprobs": obj.get("logprobs"),
        "prompt": obj.get("prompt"),
    }


def execute_tasks_save_model_hf_or_local(list_of_batch_names: list, experiment_path: str, run_prefix: str, batch_size=5) -> list:
    list_of_job_names = []

    # The model is loaded once and reused for every batch file that uses the same
    # model + LoRA. Loading it again for each file would put a second 60 GB copy
    # on the GPU and run out of memory.
    model = None
    tokenizer = None
    loaded_key = None

    for index, file_name in enumerate(list_of_batch_names):
        jsonl_file_path = f"{experiment_path}/batches/{file_name}"

        # Read configuration from the first entry of the JSONL file
        with jsonlines.open(jsonl_file_path, "r") as reader:
            for obj in reader:
                first = parse_batch_line(obj)
                model_name = first["model"]
                ft_dir = first["ft_dir"]  # optional LoRA directory
                temperature = first["temperature"]
                response_logprobs = first["response_logprobs"]
                logprobs = first["logprobs"]
                break

        if (model_name, ft_dir) != loaded_key:
            # Free the previous model (if any) before loading a different one
            if model is not None:
                print("Unloading previous model")
                model = None
                base_model = None
                gc.collect()
                torch.cuda.empty_cache()

            # Load base model (all layers on the single GPU)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": 0},
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
                # Fold the LoRA weights into the base model: same outputs, faster generation
                model = model.merge_and_unload()
                model.eval()
                print("LoRA merged into base model")
                tokenizer = AutoTokenizer.from_pretrained(ft_dir_final)
            else:
                model = base_model
                print(f"Using base model {model_name} (no LoRA applied)")
                tokenizer = AutoTokenizer.from_pretrained(model_name)

            loaded_key = (model_name, ft_dir)
        else:
            print(f"Reusing already loaded model for {file_name}")

        # Settings for batched generation with a decoder-only model
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # The model's own stop tokens (for chat models this includes end-of-turn)
        model_eos = model.generation_config.eos_token_id
        if model_eos is None:
            model_eos = []
        elif not isinstance(model_eos, list):
            model_eos = [model_eos]
        stop_ids = [t for t in dict.fromkeys(model_eos + [tokenizer.eos_token_id]) if t is not None]

        if ft_dir:
            # The LoRA was trained on "User: <prompt>\nAssistant: <answer>\n" without an
            # EOS token, so use the same format and stop at the end of the answer line.
            prompt_format = "finetune"
            newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
            stop_ids.append(newline_id)
            print("Prompt format: 'User: <prompt>\\nAssistant:' (stops at newline)")
        else:
            prompt_format = "chat" if ensure_chat_template(tokenizer, model_name) else "plain"
            print(f"Prompt format: {prompt_format}")

        def build_input(prompt):
            if prompt_format == "finetune":
                return f"User: {prompt}\nAssistant:"
            if prompt_format == "chat":
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return prompt

        # Chat templates already contain the BOS token; the other formats need it added
        add_special_tokens = prompt_format != "chat"
        stop_set = set(stop_ids)
        top_k = logprobs or 5
        do_sample = bool(temperature) and temperature > 0
        gen_kwargs = dict(
            max_new_tokens=200,
            do_sample=do_sample,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=bool(response_logprobs),
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        @torch.inference_mode()
        def run_batch(batch_inputs, batch_ids, f_out):
            enc = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                add_special_tokens=add_special_tokens,
            ).to(model.device)
            out = model.generate(**enc, **gen_kwargs)

            prompt_len = enc["input_ids"].shape[1]
            new_tokens = out.sequences[:, prompt_len:]
            # out.scores: tuple with one [batch, vocab] tensor per generated step
            scores = out.scores if gen_kwargs["output_scores"] else None

            for i in range(new_tokens.shape[0]):
                custom_id = batch_ids[i]
                ids = new_tokens[i].tolist()
                # Length of this answer: up to and including its first stop token
                n = len(ids)
                for j, tok in enumerate(ids):
                    if tok in stop_set:
                        n = j + 1
                        break
                gen_text = tokenizer.decode(ids[:n], skip_special_tokens=True).strip()
                entries = (
                    logprob_entries_from_scores(
                        torch.stack([step[i] for step in scores[:n]]), tokenizer, top_k
                    )
                    if scores is not None and n > 0 else []
                )
                json_line = format_results_huggingface(gen_text, entries, custom_id)
                f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                f_out.flush()

        # Prepare output file (named after the batch number, not the execution order)
        batch_num = re.search(r"_batch_(\d+)_", file_name).group(1)
        date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
        os.makedirs(f"{experiment_path}/results", exist_ok=True)
        output_file = f"{experiment_path}/results/{run_prefix}_results_batch_{batch_num}_{date_string}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f_out:
            with jsonlines.open(jsonl_file_path, "r") as reader:
                batch_messages = []
                batch_ids = []

                # Process prompts in batches
                for obj in tqdm(reader, desc=f"Processing {jsonl_file_path}"):
                    line = parse_batch_line(obj)
                    if not line["custom_id"]:
                        raise ValueError(f"{file_name}: batch line without custom_id, ids would not be unique across batches")
                    batch_messages.append(build_input(line["prompt"]))
                    batch_ids.append(line["custom_id"])
                    if len(batch_messages) == batch_size:
                        run_batch(batch_messages, batch_ids, f_out)
                        batch_messages = []
                        batch_ids = []

                # Process any remaining messages
                if len(batch_messages) > 0:
                    run_batch(batch_messages, batch_ids, f_out)
                    batch_messages = []
                    batch_ids = []


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
            wait_time = 150
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
                wait_time = 150
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
                client, list_of_batch_names=list_of_batch_names, experiment_path=EXPERIMENT_PATH )
            # wait for batch jobs to finish
            while not retrieve_batch_job_google(batch_job_names=list_of_job_ids):
                wait_time = 150
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