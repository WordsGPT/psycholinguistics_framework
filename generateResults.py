import json
import csv
from pathlib import Path
import numpy as np
import re
import time
import pandas as pd
import sys


""" USAGE:
python generateResults_json_output.py [EXPERIMENT_PATH][mode] [language]

modes: 
- json (output of estimations is a JSON with the word and its prediction. It checks if the word in the input matches the word in the output)
- weighted_sum (wheighted sum of the logprobs of the tokens in the word. Only valid if single token output)
- number (output of estimations is a number, the estimation of the word)

languages:
- german
"""

def extract_word_input(text):
    match = re.search(f'{open_quotations_constant}(.*?){closing_quotations_constant}', text)
    if match:
        word = match.group(1)
        return word
    return None


def extract_word_output(text):
    match = re.search(f'"{word_constant}"\s*:\s*"([^"]+)"', text)
    if match:
        word = match.group(1)
        return word
    return None


def extract_number(text):
    match = re.search(f'"{feature_constant}"\s*:\s*"([0-9]*\.?[0-9]+)"', text)
    if not match:
        match = re.search(f'"{feature_constant}"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if not match:
        all_matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+\.\d*|[-+]?\d+', text)
        if all_matches:
            return float(all_matches[-1])
    if match:
        return float(match.group(1))
    return None


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

## OpenAI ##

def openAI_processing(results_content_file, batches_content_file):
    match_key = 'custom_id'
    lookup = {entry[match_key]: entry for entry in batches_content_file if match_key in entry}
    combined_data = []
    index = 0
    for entry in results_content_file:
        entry_result = {}
        index += 1
        weighted_sum = None
        logprob = None
        if match_key in entry and entry[match_key] in lookup:
            combined_entry = {**entry, **lookup[entry[match_key]]}
            custom_id = combined_entry['custom_id']
            if mode == "json":             
                word_input = extract_word_input(combined_entry['body']['messages'][0]['content'])
                word_output = extract_word_output(combined_entry['response']['body']['choices'][0]['message']['content'])

                feature_value = extract_number(combined_entry['response']['body']['choices'][0]['message']['content'])
                if word_input and word_output:
                    if word_input != word_output:
                        print(f"Warning: custom Id: '{custom_id}. Word input '{word_input}' does not match word output '{word_output}'")
                        #feature_value = '#N/D'
            elif mode == "weighted_sum" or mode == "number":
                word_input = extract_word_input(combined_entry['body']['messages'][0]['content'])
                # Only valid for responses of single token
                if len(combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"]) == 1:
                    top_logprobs_list = combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"][0]['top_logprobs']
                    weighted_sum = 0
                    # Iterate over the list of top_logprobs that are numbers
                    for top_logprob in top_logprobs_list:
                        try:
                            token_value = int(top_logprob['token'])
                            logprob_value = top_logprob['logprob']
                            weighted_sum += token_value * np.exp(float(logprob_value))
                        except ValueError:
                            pass
                    logprob = combined_entry['response']['body']['choices'][0]['logprobs']['content'][0]['logprob']
                feature_value = combined_entry['response']['body']['choices'][0]['message']['content']

            entry_result['word'] = word_input
            # entry_result['custom_id'] = custom_id
            entry_result[feature_column] = feature_value

            if logprob is not None:
                entry_result['logprob'] = logprob
                logprobs = True
            if weighted_sum is not None:
                entry_result['weighted_sum'] = weighted_sum

            combined_data.append(entry_result)

    return combined_data

## Google ##

def google_processing(results_content_file, batches_content_file):
    """Deserialize Google (Gemini) batch/results JSONL and return rows.

    Expects structure like:
    - batches *.jsonl lines: {"key": str, "request": {"contents": [{"parts": [{"text": str}, ...]}], ...}}
    - results *.jsonl lines: {"key": str, "response": {"candidates": [{"content": {"parts": [{"text": str}]}, "logprobsResult": {...}}]}}
    """
    # Build lookup from batches by key/custom_id (prefer 'key' for Google format)
    def get_match_value(obj):
        return obj.get('key')
    
    lookup = {}
    for entry in batches_content_file:
        matck_key = get_match_value(entry)
        if matck_key is not None:
            lookup[matck_key] = entry

    combined_data = []
    for entry in results_content_file:
        matck_key = get_match_value(entry)
        if matck_key is None or matck_key not in lookup:
            continue

        batch_item = lookup[matck_key]
        result_item = entry.get('response', {})

        # Extract the prompt text (to recover the word inside quotes)
        prompt_text = None
        try:
            contents = batch_item.get('request', {}).get('contents', [])
            parts_texts = []
            for c in contents:
                for p in c.get('parts', []):
                    t = p.get('text')
                    if isinstance(t, str):
                        parts_texts.append(t)
            if parts_texts:
                prompt_text = "\n".join(parts_texts)
        except Exception:
            prompt_text = None

        word_input = extract_word_input(prompt_text) if prompt_text else None

        # Extract the model output text
        output_text = None
        try:
            candidates = result_item.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts:
                    output_text = parts[0].get('text')
        except Exception:
            output_text = None

        # Parse number from output text
        feature_value = extract_number(output_text or '')

        entry_result = {
            'word': word_input,
            feature_column: feature_value
        }

        # Optional: compute weighted_sum and logprob for single numeric token outputs
        try:
            if mode in ("weighted_sum", "number"):
                is_token_content_single_token_number = str(output_text).isdigit() and int(output_text) < 1000 # only positive integers and tokenizers until 999 with one token
                candidates = result_item.get('candidates', [])
                if candidates:
                    cand0 = candidates[0]
                    lpr = cand0.get('logprobsResult') or {}
                    chosen = lpr.get('chosenCandidates') or []
                    if chosen:
                        # Logprob of the chosen first token (likely the digit)
                        entry_result['logprob'] = chosen[0].get('logProbability')
                    top_list = lpr.get('topCandidates') or []

                    if is_token_content_single_token_number: 
                        # Only use the first step (the numeric token) for weighted sum
                        top_first = top_list[0]
                        weighted_sum = 0.0
                        for tc in top_first.get('candidates', []):
                            tok = tc.get('token')
                            lp = tc.get('logProbability')
                            try:
                                tok_val = int(tok)
                                weighted_sum += tok_val * float(np.exp(float(lp)))
                            except Exception:
                                pass
                        entry_result['weighted_sum'] = weighted_sum
        except Exception:
            # Make logprob/weighted_sum optional; ignore failures silently
            pass

        combined_data.append(entry_result)

    return combined_data


## HuggingFace ##

def huggingface_processing(results_content_file, batches_content_file):
    match_key = 'id'
    lookup = {entry[match_key]: entry for entry in batches_content_file if match_key in entry}
    combined_data = []
    index = 0
    for entry in results_content_file:
        entry_result = {}
        index += 1
        weighted_sum = None
        logprob = None
        if match_key in entry and entry[match_key] in lookup:
            combined_entry = {**entry, **lookup[entry[match_key]]}
            custom_id = combined_entry[match_key]
            if mode == "json":             
                word_input = extract_word_input(combined_entry['prompt'])
                word_output = extract_word_output(combined_entry['response']['body']['choices'][0]['message']['content'])

                feature_value = extract_number(combined_entry['response']['body']['choices'][0]['message']['content'])
                if word_input and word_output:
                    if word_input != word_output:
                        print(f"Warning: custom Id: '{custom_id}. Word input '{word_input}' does not match word output '{word_output}'")
                        #feature_value = '#N/D'
            elif mode == "weighted_sum" or mode == "number":
                word_input = extract_word_input(combined_entry['prompt'])
                feature_value = combined_entry['response']['body']['choices'][0]['message']['content']
                is_token_content_single_token_number = str(feature_value).isdigit() and int(feature_value) < 1000
                # Only valid for responses of single util tokens
                if is_token_content_single_token_number:
                    top_logprobs_list = combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"][0]['top_logprobs']
                    weighted_sum = 0
                    # Iterate over the list of top_logprobs that are numbers
                    for top_logprob in top_logprobs_list:
                        try:
                            token_value = int(top_logprob['token'])
                            logprob_value = top_logprob['logprob']
                            weighted_sum += token_value * np.exp(float(logprob_value))
                        except ValueError:
                            pass
                    logprob = combined_entry['response']['body']['choices'][0]['logprobs']['content'][0]['logprob']
                

            entry_result['word'] = word_input
            # entry_result['custom_id'] = custom_id
            entry_result[feature_column] = feature_value

            if logprob is not None:
                entry_result['logprob'] = logprob
                logprobs = True
            if weighted_sum is not None:
                entry_result['weighted_sum'] = weighted_sum

            combined_data.append(entry_result)

    return combined_data

## Main ##

if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]
        if len(sys.argv) > 3:
            country = sys.argv[3]
        else:
            country = ""
    else:
        print(
            "Provide as arguments the experiment path, the mode [json, number, weighted_sum] and optionally the language."
        )
        exit()

    open_quotations_constant = '"'
    closing_quotations_constant = '"'
    # TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
    word_constant = 'Word'
    feature_column = 'familiarity'
    feature_constant = 'AoA'
    # END TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
    logprobs = False
    timestamp = int(time.time())
    output_file = f'{EXPERIMENT_PATH}/output_{timestamp}.xlsx'


    if country == 'german':
        # TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
        word_constant = 'Wort'
        feature_constant = 'Erwerbsalter'
        # END TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
        open_quotations_constant = '„'
        closing_quotations_constant = '”'


    results_file = f"{EXPERIMENT_PATH}/results/results.jsonl"
    batches_file = f"{EXPERIMENT_PATH}/batches/batches.jsonl"


    results_content_file = read_jsonl(results_file)
    batches_content_file = read_jsonl(batches_file)

    if 'custom_id' in batches_content_file[0]:
        combined_data = openAI_processing(results_content_file, batches_content_file)
    elif 'key' in batches_content_file[0]:
        combined_data = google_processing(results_content_file, batches_content_file)
    elif 'id' in batches_content_file[0]:
        combined_data = huggingface_processing(results_content_file, batches_content_file)
    else:
        print("Unknown batch format, cannot process results.")

    all_fieldnames = list(combined_data[0].keys())

    df = pd.DataFrame(combined_data)
    df.to_excel(output_file, index=False, columns=all_fieldnames)

    print(f"Combined data written to {output_file}")

