# Iconicity estimates for Spanish and English words

This repository contains code, input word lists, human-rated normative data and AI-generated iconicity estimates for the associated paper. The main datasets cover **65,305 English entries** and **127,727 Spanish estimates**. It is organized by language rather than by unrelated psycholinguistic experiments.

## Main files at a glance

| Language | Input word list | Human ratings | Fine-tuning train/test | Published output | Joined human/model results |
|---|---|---|---|---|---|
| English | [`iconicity_english/data/List_English-Words.xlsx`](iconicity_english/data/List_English-Words.xlsx) | [`iconicity_english/data/iconicity_ratings_english.xlsx`](iconicity_english/data/iconicity_ratings_english.xlsx) | [`iconicity_english/finetuning/`](iconicity_english/finetuning/) | [`iconicity_english/output/iconicity_BIGFILE_ratings_english.xlsx`](iconicity_english/output/iconicity_BIGFILE_ratings_english.xlsx) | [`iconicity_english/output/iconicity_ratings_english_joined.xlsx`](iconicity_english/output/iconicity_ratings_english_joined.xlsx) |
| Spanish | [`iconicity_spanish/data/List-Spanish-words.xlsx`](iconicity_spanish/data/List-Spanish-words.xlsx) | [`iconicity_spanish/data/iconicity_ratings_spanish.xlsx`](iconicity_spanish/data/iconicity_ratings_spanish.xlsx) | [`iconicity_spanish/finetuning/`](iconicity_spanish/finetuning/) | [`iconicity_spanish/output/iconicity_BIGFILE_ratings_spanish.xlsx`](iconicity_spanish/output/iconicity_BIGFILE_ratings_spanish.xlsx) | [`iconicity_spanish/output/iconicity_ratings_spanish_joined.xlsx`](iconicity_spanish/output/iconicity_ratings_spanish_joined.xlsx) |

The `*_BIGFILE_*` spreadsheets contain two columns: `word` and model-estimated `iconicity` (1–7). The `*_joined.xlsx` files combine the human judgments with base-model and fine-tuned-model predictions. The full column dictionaries are in each language's [`data/`](iconicity_english/data/) and [`output/`](iconicity_english/output/) README files (and the corresponding Spanish folders).

**Known difference:** the Spanish input list has 127,728 entries, but the released BIGFILE has 127,727; the missing entry is `fuisca` (row 59835 of the input workbook). It has no generated estimate in this snapshot.

## Repository contents

| Directory or file | Purpose |
|---|---|
| [`.gitignore`](.gitignore) | Patterns for generated files, local environments and untracked secrets. |
| [`.vscode/`](.vscode/) | Developer editor settings (not part of the published dataset). |
| [`apis_example.env`](apis_example.env) | Template of API environment variable names; fill a private `apis.env` locally (never publish actual credentials). |
| [`combine_excels.py`](combine_excels.py) | Joins two Excel files by selected columns; inspect arguments before use because it can replace the first file after making a backup. |
| [`config_example.yaml`](config_example.yaml) | Generic example configuration; includes other providers and should not be mistaken for the iconicity experiment configurations. |
| [`create_finetuning_dataset.py`](create_finetuning_dataset.py) | Splits human-rated entries and creates the JSONL training input for fine-tuning. |
| [`execute_experiment.py`](execute_experiment.py) | Submits and retrieves model batch experiments defined in `config.yaml`. |
| [`execute_finetune.py`](execute_finetune.py) | Launches a fine-tuning job using the specified experiment configuration. |
| [`execute_individual_api_google.py`](execute_individual_api_google.py) | Runs individual requests through Google/Gemini API. |
| [`execute_individual_api_google_finetuned.py`](execute_individual_api_google_finetuned.py) | Runs individual queries with a fine-tuned Google model. |
| [`execute_indiviual_api_openai.py`](execute_indiviual_api_openai.py) | Runs individual requests through the OpenAI API (filename retains original spelling). |
| [`execute_indiviual_huggingface.py`](execute_indiviual_huggingface.py) | Runs individual requests using Hugging Face APIs (filename retains original spelling). |
| [`execute_indiviual_local.py`](execute_indiviual_local.py) | Runs model requests in a local execution setup (filename retains original spelling). |
| [`generateResults.py`](generateResults.py) | General script for converting batch responses to tabular estimates (numeric, JSON or weighted-score modes). |
| [`generateResults_iconicity.py`](generateResults_iconicity.py) | Iconicity-oriented variant of result extraction and score processing; verify which run created each released output. |
| [`generateResults_vertex.py`](generateResults_vertex.py) | Transforms Google Vertex/Gemini response records into tabular results. |
| [`iconicity_english/`](iconicity_english/) | English experiment: configuration, source word lists, training splits, prompts and scored outputs. |
| [`iconicity_spanish/`](iconicity_spanish/) | Spanish experiment: configuration, source word lists, training splits, prompts, saved responses and scored outputs. |
| [`LICENSE`](LICENSE) | Original root Apache License 2.0 text; retained unchanged. |
| [`LICENSE-APACHE-2.0.txt`](LICENSE-APACHE-2.0.txt) | Full original legal text of Apache License 2.0 (local copy). |
| [`LICENSE-CC0-1.0.txt`](LICENSE-CC0-1.0.txt) | Full original legal code for CC0 1.0 Universal (local copy). |
| [`LICENSE-DATA.md`](LICENSE-DATA.md) | CC0 1.0 dedication and scope for the data files; source code remains under Apache 2.0. |
| [`new_batches_file_with_no_processed_lines.py`](new_batches_file_with_no_processed_lines.py) | Utility to prepare batch requests excluding already processed items; **CHECK** exact input/output naming in script before reuse. |
| [`prepare_and_execute_gemini_experiment.py`](prepare_and_execute_gemini_experiment.py) | Helper for preparing and running Gemini experiments. |
| [`prepare_experiment.py`](prepare_experiment.py) | Builds request batches from a configured word column and prompt. |
| [`prepare_experiment_vertex_gemini3.py`](prepare_experiment_vertex_gemini3.py) | Creates batch jobs for Gemini via Google Vertex AI. |
| [`psycholinguistics_framework.zip`](psycholinguistics_framework.zip) | Existing ZIP snapshot of code/materials; **CHECK:** whether it is still needed or redundant with this repository. |
| [`requirements.txt`](requirements.txt) | Python package dependencies with versions for reproducing execution. |
| [`requirements_no_version.txt`](requirements_no_version.txt) | Python dependency names without pinned versions. |
| [`utils.py`](utils.py) | Shared helpers for model API connections, configuration loading and data processing. |

## Reproducing the experiment (overview)

1. Use Python version 3.11 or above (checked up to version 3.14.10, use requirements_no_version in newer Pythons) according to the original repository instructions; install dependencies with `pip install -r requirements.txt`.
2. Copy `apis_example.env` to a private local `apis.env` and supply your own credentials; **never commit or publish real keys**.
3. Read `iconicity_english/config.yaml` or `iconicity_spanish/config.yaml` (documented in the respective language README). Each defines base, fine-tuned and large-word-list runs.
4. Prepare batches with `python3 prepare_experiment.py <experiment_folder> <experiment_name>`. Execute them with `python3 execute_experiment.py <experiment_folder> <experiment_name>`.
5. For supervised learning, use `python3 create_finetuning_dataset.py <experiment_folder> <fine_tuning_name>` followed by `python3 execute_finetune.py <experiment_folder> <fine_tuning_name>`. The train/test files have a 50/50 split and random state 42.
6. Batch response records are processed using `generateResults_iconicity.py`.

Example run names are defined exactly in the language-specific config files; placeholders above are arguments, not literal paths. These instructions outline the original workflow but are **not** yet a fully audited end-to-end recipe. The API may produce different results if the hosted model or endpoint changes.

## Configuration fields

The two experiment READMEs document every configured experiment key, input dataset, column, prompt, base/fine-tuned model ID and the fine-tuning split/target.

## Input provenance

- English large vocabulary: Brysbaert, Martínez & Reviriego (2025), Moving beyond word frequency based on tally counting; source data: https://osf.io/c2yef/files/bafpv
- Spanish large vocabulary: Martínez et al. (2025), AI-generated estimates of familiarity, concreteness, valence, and arousal for over 100,000 Spanish words; source data: https://osf.io/frc6a/files/9672j
- English human ratings: Winter et al. (2024), Iconicity ratings for 14,000+ English words: https://doi.org/10.3758/s13428-023-02112-6
- Spanish human ratings: Hinojosa et al. (2021), Iconicity ratings for 10,995 Spanish words: https://doi.org/10.3758/s13428-020-01496-z

The original word entries have been retained to support comparison and alignment with previous norming datasets; they can include numerals, punctuation variants and unconventional expressions. These entries may require special care when interpreting iconicity.

## Licensing

- **Data:** [CC0 1.0 Universal](LICENSE-DATA.md) (full [CC0 legal text](LICENSE-CC0-1.0.txt)), subject to the rights actually held in the released data.
- **Software:** [Apache License 2.0](LICENSE) (additional [Apache legal-text copy](LICENSE-APACHE-2.0.txt)).

When archiving the datasets in Zenodo, display CC0 on the dataset landing page. **TO COMPLETE:** add the dataset DOI (https://doi.org/...) and version-specific citation once assigned.

## Other notes

- The source repository may produce temporary `batches/`, `results/`, `__pycache__/` and other intermediate folders while scripts run. They are not required to interpret the released XLSX datasets.
- This distributable archive intentionally excludes the local credential-bearing `apis.env`, Python bytecode caches and `.git` metadata; generate credentials locally from `apis_example.env`.