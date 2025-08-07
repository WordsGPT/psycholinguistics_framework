import json
import csv
from pathlib import Path
import numpy as np
import re
import time
import pandas as pd
import sys


""" USAGE:
python generateResults_json_output.py [mode] [language]

modes: 
- json (output of estimations is a JSON with the word and its prediction. It checks if the word in the input matches the word in the output)
- weighted_sum (wheighted sum of the logprobs of the tokens in the word. Only valid if single token output)
- number (output of estimations is a number, the estimation of the word)

languages:
- german
"""

mode = sys.argv[1] if len(sys.argv) > 1 else "json"
country = sys.argv[2] if len(sys.argv) > 2 else None

word_constant = 'Word'
open_quotations_constatnt = '"'
closing_quotations_constant = '"'
feature_column = 'aoa'
feature_constant = 'AoA'
logprobs = False
timestamp = int(time.time())
output_file = f'output_{timestamp}.xlsx'


if country == 'german':
    word_constant = 'Wort'
    open_quotations_constatnt = '„'
    closing_quotations_constant = '”'
    feature_constant = 'Erwerbsalter'


def extract_word_input(text):
    match = re.search(f'{open_quotations_constatnt}(.*?){closing_quotations_constant}', text)
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
    if match:
        return float(match.group(1))
    match = re.search(f'"{feature_constant}"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if match:
        return float(match.group(1))
    return None


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


jsonl_file_1 = "results/results.jsonl"
jsonl_file_2 = 'batches/batches.jsonl'


data1 = read_jsonl(jsonl_file_1)
data2 = read_jsonl(jsonl_file_2)
print('loaded')
match_key = 'custom_id' # Change this to the relevant field for matching

lookup = {entry[match_key]: entry for entry in data2 if match_key in entry}


combined_data = []

index = 0
for entry in data1:
    entry_result = {}
    index += 1
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
            weighted_sum = None
            logprob = None
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
            if logprob is not None:
                entry_result['logprob'] = logprob
                logprobs = True
            if weighted_sum is not None:
                entry_result['weighted_sum'] = weighted_sum

        entry_result['custom_id'] = custom_id
        entry_result['word'] = word_input
        entry_result[feature_column] = feature_value

        combined_data.append(entry_result)

all_fieldnames = list(combined_data[0].keys()) 

# Guardar también como Excel
df = pd.DataFrame(combined_data)
df.to_excel(output_file, index=False, columns=all_fieldnames)

print(f"Combined data written to {output_file}")

