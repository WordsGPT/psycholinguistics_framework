# Spanish iconicity — experiment

This folder belongs to the iconicity estimation experiments. The files listed below are those present in this repository snapshot.

## Files

| File | Purpose |
|---|---|
| [`config.yaml`](config.yaml) | Defines the named base, fine-tuned and extended-word-list runs for this language, including model IDs and dataset/prompt paths. |

## Configuration (`config.yaml`) — B06

The configuration has separate sections for `experiments` (prediction runs) and `finetuning` (supervised model training). The values below are read from the accompanying YAML; no default values have been invented.

| Field | Meaning |
|---|---|
| `experiments.<name>` | Unique experiment identifier used by the command-line scripts. |
| `dataset_path` | Input XLSX filename, located in this experiment's `data/` directory. |
| `dataset_column` | Exact header to read (`Word` for extended lists, `word` for human norms/splits). |
| `prompt_path` | Text template in `prompts/` used to generate the model input. |
| `model_name` | Exact model identifier supplied to the API; includes specific fine-tuned model IDs for tuned runs. |
| `company` | API/provider (`OpenAI` in these experiment definitions). |
| `finetuning.<name>` | Fine-tuning job definition; not a prediction run. |
| `ft_dataset_path` | Input human-rated dataset to split into training and validation records. |
| `train_split_percentage` | Fraction of input rows assigned to training (0.5). |
| `random_state` | Random seed used in dataset division (42). |
| `answer_column` | Human-response target (`rating` for English; `ico-m` for Spanish). |
| `dataset_finetune_name` | Filename of prepared JSONL fine-tuning records. |
| `especial_suffix` | Model-run suffix used when submitting the fine-tuning task (spelling retained from code). |

### Configured runs

| Experiment key | Dataset | Model |
|---|---|---|
| `iconicity_spanish_base_prompt_gpt-4o` | `iconicity_ratings_spanish.xlsx` | `gpt-4o-mini-2024-07-18` |
| `iconicity_spanish_base_prompt_gpt-4o_finetuned` | `iconicity_ratings_spanish.xlsx` | `ft:gpt-4o-mini-2024-07-18:ging-upm:iconicity-spanish-base-prompt-gpt-4o:DZEwYFcF` |
| `iconicity_spanish_big_list_gpt-4o_finetuned` | `List-Spanish-words.xlsx` | `ft:gpt-4o-mini-2024-07-18:ging-upm:iconicity-spanish-base-prompt-gpt-4o:DZEwYFcF` |

### Configuration points requiring confirmation

- **TO CONFIRM:** the API endpoint and request parameters actually used for each batch and fine-tuning job (`temperature`, `top_logprobs`, `max_tokens`, seed where applicable); these are not fully recorded in `config.yaml`.
- **TO CONFIRM:** exact processing script, mode (`number` / `weighted_sum`), and mapping of a completed API response to the released `BIGFILE` output.
- **TO CONFIRM:** original training-job metadata (epochs, batch size, LR multiplier, job identifier, date), which cannot be inferred from these YAML fields alone.
- **TO CONFIRM:** specific steps and script invocation used to merge the model results into the `*_joined.xlsx` file.
- **TO CONFIRM:** any non-default API settings and retry/failed-response policy, including handling of missing predictions.

**Naming note:** the experiment keys contain `gpt-4o`, but the base model in the YAML is explicitly `gpt-4o-mini-2024-07-18`. Do not treat these as interchangeable model names.


**Known list/output difference:** `data/List-Spanish-words.xlsx` has 127,728 entries; `output/iconicity_BIGFILE_ratings_spanish.xlsx` has 127,727. The missing entry is **`fuisca`** (input spreadsheet row 59835). No prediction was invented.
