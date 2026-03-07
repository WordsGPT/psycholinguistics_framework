"""
This script is designed to execute Gemini tasks for psycholinguistic experiments, one by one.
"""

import json
import os
import sys
from   datetime import datetime

import pandas as pd

from google import genai
from google.genai import types

from utils import load_config, google_login, read_txt

def load_word_list(file_path: str, column_name: str) -> list:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding='iso-8859-1')
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
    word_list = df[column_name].tolist()
    print(f"Successfully loaded {len(word_list)} words from {file_path}.")
    return word_list


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            EXPERIMENT_NAME = sys.argv[2]
            if len(sys.argv) > 3:
               GET_LOGPROBS = sys.argv[3]
               if GET_LOGPROBS != 'True':
                  GET_LOGPROBS = False
    
            else: GET_LOGPROBS = True
        else:
            EXPERIMENT_NAME = "original"
    else:
        print( "Provide as arguments the experiment path and optionally the experiment name and Logprobs retrieval switch.\n"
               "i.e.: python3 prepare_experiment.py <EXPERIMENT_PATH> <EXPERIMENT_NAME>." )
        exit()

    print(f'EXPERIMENT_PATH: {EXPERIMENT_PATH}\nEXPERIMENT_NAME: {EXPERIMENT_NAME}\nGET_LOGPROBS: {GET_LOGPROBS}')

    # Get experiment configuration:
    config_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="experiments",
        name=EXPERIMENT_NAME,
    )

    # Check valid company ("Google"):
    company = config_args["company"]
    if company != "Google":
        raise ValueError(f"Company not implemented: {company}")

    # Get queries words from Excel file
    word_list = load_word_list(
        file_path=f"{EXPERIMENT_PATH}/data/{config_args['dataset_path']}",
        column_name=config_args.get("dataset_column", "word"),
    )

    # Get prompt template:
    prompt = read_txt(
        file_path=f"{EXPERIMENT_PATH}/prompts/{config_args['prompt_path']}"
    )

    # Build prompts:
    prompt_key = f"{{{config_args["dataset_column"]}}}"
    prompts = [prompt.replace(prompt_key, word) for word in word_list]

    # Build input for client.models.generate_content
    contents = [
        [types.Content(role="user", parts=[types.Part.from_text(text = query)])]
        for query in prompts
    ]

    if GET_LOGPROBS:
        content_config = types.GenerateContentConfig(
            temperature = 0,
            # No thinking:
            #thinking_config = types.ThinkingConfig(thinking_budget = 0),
            # Thinking active (Gemini 2.5 and above)
            #thinking_config=types.ThinkingConfig(include_thoughts=True),
            response_logprobs = True,
            logprobs = 5
            )
    else:
        content_config = types.GenerateContentConfig(
            temperature = 0,
            # No logprobs (currently not suported in Gemini >= 2.5)
            )

    model = config_args["model_name"]
    client = google_login()

###############################################################################

    # Results list to dump to excel file:
    results = list()

    # Remove ... from prompt_key:
    prompt_key_clean = prompt_key.replace('{', '').replace('}', '')
    
    # Build the results column name:
    experiment_name_underscore_pos = EXPERIMENT_NAME.find('_')
    experiment_name_prefix = EXPERIMENT_NAME
    if experiment_name_underscore_pos > 1:
        experiment_name_prefix = EXPERIMENT_NAME[0:experiment_name_underscore_pos]

    for index, word in enumerate(word_list[81:82]) :

        experiment_output = client.models.generate_content(
           model = model, contents = contents[index], config = content_config
        )
        
        # Debug looking for returned fields
        # breakpoint()

        # Get experiment_value from experiment_output
        experiment_value = 'N/D'
        try:
           experiment_value = int(experiment_output.text[0])
        except ValueError:
           print( f"Unexpected answer to prompt: {experiment_output.parts[0]}" )

        if GET_LOGPROBS:
            logprop_results = experiment_output.candidates[0].\
                logprobs_result.top_candidates[0].candidates
            logprob_value = 0
            row = { f"{prompt_key_clean}": f"{word}", \
                    f"{experiment_name_prefix}": experiment_value, \
                    f"{experiment_name_prefix}_logprob": logprob_value }
        else:
            row = { f"{prompt_key_clean}": f"{word}", \
                    f"{experiment_name_prefix}": experiment_value }
        
        results.append( row )
        print( index, row, flush = True )

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_file = output_file = f'{EXPERIMENT_PATH}/output/{EXPERIMENT_NAME}_{timestamp}.xlsx'
    df.to_excel(output_file, index=False)
    print( f"Written: {output_file}" )
