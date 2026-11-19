# English iconicity — output

This folder belongs to the iconicity estimation experiments. The files listed below are those present in this repository snapshot.

## Files

| File | Purpose |
|---|---|
| [`correlations.py`](correlations.py) | Python script to compute Pearson correlation matrices from the joined human/model-rating table. |
| [`correlations_all_local.xlsx`](correlations_all_local.xlsx) | Correlation matrix for English human ratings and base/fine-tuned model outputs. |
| [`correlations_test_dataset.xlsx`](correlations_test_dataset.xlsx) | Correlation matrix calculated for the English test subset. |
| [`correlations_train_dataset.xlsx`](correlations_train_dataset.xlsx) | Correlation matrix calculated for the English training subset. |
| [`correlations_upm-llm.xlsx`](correlations_upm-llm.xlsx) | Pearson correlation matrix between English human ratings, base/fine-tuned outputs and UPM-LLM. |
| [`iconicity_BIGFILE_ratings_english.xlsx`](iconicity_BIGFILE_ratings_english.xlsx) | Main published output: words/expressions and their LLM-estimated iconicity scores. |
| [`iconicity_ratings_english_joined.xlsx`](iconicity_ratings_english_joined.xlsx) | Combined human iconicity norms and base/fine-tuned model predictions; main validation/comparison table. |
| [`test_split_joined.xlsx`](test_split_joined.xlsx) | English human-rated held-out split enriched with model-generated ratings (`iconicity_new`). |
| [`train_split_joined.xlsx`](train_split_joined.xlsx) | English human-rated training split enriched with model-generated ratings (`iconicity_new`). |

Model predictions and Pearson-correlation results are not interchangeable with the human judgments from the `data/` folder.


### Data dictionary — `iconicity_BIGFILE_ratings_english.xlsx`

Rows (excluding header): **65,305**. Columns: **2**.

| Column | Meaning |
|---|---|
| `word` | Lexical entry or expression as represented in this table; case and punctuation may be meaningful. |
| `iconicity` | Model-generated iconicity estimate on the 1–7 rating scale. |

**Interpretation:** Model-generated scores are predictions, not direct human measurements. Numeric and punctuation-bearing entries were intentionally preserved to allow alignment with the input lists.


### Data dictionary — `iconicity_ratings_english_joined.xlsx`

Rows (excluding header): **14,776**. Columns: **9**.

| Column | Meaning |
|---|---|
| `word` | Lexical entry or expression as represented in this table; case and punctuation may be meaningful. |
| `n_ratings` | Number of ratings recorded in the English source dataset. |
| `n` | Count field retained from the English source norms. |
| `prop_known` | Proportion of participants indicating that they knew the word; inherited from the English norms (usually between 0 and 1). |
| `rating` | Mean human iconicity rating on a 1–7 scale (English norms; Winter et al., 2024). |
| `rating_sd` | Standard deviation of the human iconicity ratings (English normative study). |
| `iconicity_english_base_prompt_gpt-4o` | Model-generated English iconicity estimate from the base (non-fine-tuned) configuration; configuration lists `gpt-4o-mini-2024-07-18`. |
| `iconicity_english_base_prompt_gpt-4o_finetuned` | Model-generated English iconicity estimate from the fine-tuned configuration defined in `config.yaml`. |
| `iconicity_english_UPM-LLM` | Additional model-generated English estimate labelled UPM-LLM. |
