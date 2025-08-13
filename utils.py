import json
import os

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from huggingface_hub import login

def openai_login():
    load_dotenv("apis.env")
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client

def google_login():
    load_dotenv("apis.env")
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client

    
def huggingface_login():
    load_dotenv("apis.env")
    api_key = os.getenv("HUGGINGFACE_TOKEN")
    login(token=api_key)


def read_yaml(file_path: str):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def read_txt(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def load_config(experiment_path: str, config_type: str, name: str) -> dict:
    """
    Load configuration for a given experiment or fine-tuning model.

    :param experiment_path: Path to the experiment directory.
    :param config_type: Type of configuration to load ('experiments' or 'finetuning').
    :param name: Name of the experiment or fine-tuning model.
    :return: Configuration dictionary for the specified name.
    """
    config = read_yaml(file_path=f"{experiment_path}/config.yaml")

    if config_type not in config:
        print(f"Configuration type {config_type} not found in config.yaml.")
        exit()

    if name not in config[config_type]:
        print(
            f"{config_type.capitalize()} {name} not found in config.yaml. "
            f"Available {config_type}: {config[config_type].keys()}"
        )
        exit()

    config_args = config[config_type][name]
    return config_args


def read_column_as_list(file_path: str, column_name: str) -> list[int]:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return df[column_name].tolist()


def get_answers_from_results_jsonl(file_path: str) -> list[int]:
    results = []
    with open(file_path, "r") as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(
                json_object["response"]["body"]["choices"][0]["message"]["content"]
            )
    return results
