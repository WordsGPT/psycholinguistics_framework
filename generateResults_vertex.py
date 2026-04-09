import json
import re
import time
import pandas as pd
import sys
import os

"""Extracts data from a Google Vertex Batch Inference file, returned from a tuned model,
normally stored in a Google Cloud bucket.

USAGE:

python generateResults_vertex.py <experiment-path>
"""

feature_column = 'familiarity'
feature_constant = 'AoA'

def recent_file(dir):
    """
    Devuelve el nombre del fichero más reciente dentro de un directorio.
    
    :param directorio: Ruta al directorio
    :return: Nombre del fichero más reciente o None si no hay ficheros
    """

    try:
        # Obtener lista de rutas completas
        rutas = [os.path.join(dir, f) for f in os.listdir(dir)]
        
        # Filtrar solo ficheros (no directorios)
        ficheros = [f for f in rutas if os.path.isfile(f)]
        
        if not ficheros:
            return None
        
        # Obtener el fichero más reciente según la fecha de modificación
        fichero_reciente = max(ficheros, key=os.path.getmtime)
        
        # Devolver solo el nombre, no la ruta completa
        return os.path.basename(fichero_reciente)
    
    except Exception as e:
        print(f"Error: {e}")
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


def google_processing(results_content_file):
    """Deserialize Google (Gemini) batch/results JSONL and return rows.

    Expects structure like:
    - results *.jsonl lines: {"key": <key>, "request": ..., "status": ..., "response":{"candidates":[{"avgLogprobs": ..., "content": {"parts":[{"text": <familiarity> }], ...}
    """
    # Build lookup from batches by key/custom_id (prefer 'key' for Google format)

    extracted_data = []
 
    for entry in results_content_file:
        key = entry["key"]

        result_item = entry.get('response', {})

        # Extract the model output text
        output_text = None
        try:
            candidates = result_item.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts:
                    output_text = parts[0].get('text')
                # Parse number from output text
                feature_value = extract_number(output_text or '')
        except Exception:
            output_text = None

        entry_result = {
            'Word': key,
            feature_column: feature_value
        }

        extracted_data.append(entry_result)

    extracted_data.sort(key=lambda x: x["Word"].lower())
    return extracted_data

## Main ##

if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
    else:
        print("Provide as arguments the experiment path")
        exit()

    results_file = recent_file(f"{EXPERIMENT_PATH}/results_vertex")
    output_file = f'{EXPERIMENT_PATH}/output_vertex/output_{int(time.time())}.xlsx'

    results_content = read_jsonl(f"{EXPERIMENT_PATH}/results_vertex/{results_file}")

    if 'key' in results_content[0]:
        extracted_data = google_processing(results_content)
    else:
        print("Unknown batch format, cannot process results.")

    df = pd.DataFrame(extracted_data)
    df.to_excel(output_file, index=False)

    print(f"Extracted data written to {output_file}")
