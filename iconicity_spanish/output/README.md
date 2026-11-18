# Spanish iconicity — output

This folder belongs to the iconicity estimation experiments. The files listed below are those present in this repository snapshot.

## Files

| File | Purpose |
|---|---|
| [`correlations.py`](correlations.py) | Python script to compute Pearson correlation matrices from the joined human/model-rating table. |
| [`correlations.xlsx`](correlations.xlsx) | Pearson correlation matrix between Spanish human ratings and two model estimates. |
| [`iconicity_BIGFILE_ratings_spanish.xlsx`](iconicity_BIGFILE_ratings_spanish.xlsx) | Main published output: words/expressions and their LLM-estimated iconicity scores. |
| [`iconicity_ratings_spanish_joined.xlsx`](iconicity_ratings_spanish_joined.xlsx) | Combined human iconicity norms and base/fine-tuned model predictions; main validation/comparison table. |

Model predictions and Pearson-correlation results are not interchangeable with the human judgments from the `data/` folder.


### Data dictionary — `iconicity_BIGFILE_ratings_spanish.xlsx`

Rows (excluding header): **127,727**. Columns: **2**.

| Column | Meaning |
|---|---|
| `word` | Lexical entry or expression as represented in this table; case and punctuation may be meaningful. |
| `iconicity` | Model-generated iconicity estimate on the 1–7 rating scale. **CHECK:** confirm exact aggregation and handling of invalid/missing responses in the generating script and output provenance. |

**Interpretation:** Model-generated scores are predictions, not direct human measurements. Numeric and punctuation-bearing entries were intentionally preserved to allow alignment with the input lists.


### Data dictionary — `iconicity_ratings_spanish_joined.xlsx`

Rows (excluding header): **10,995**. Columns: **12**.

| Column | Meaning |
|---|---|
| `word` | Lexical entry or expression as represented in this table; case and punctuation may be meaningful. |
| `ico-m` | Mean human iconicity rating in the visually presented Spanish norms, scale 1–7. |
| `ico-sd` | Standard deviation of Spanish human iconicity ratings (visual presentation). |
| `ico-n` | Number of participants who rated Spanish visual iconicity. |
| `ico-dn` | Number of participants who indicated that they did not know the Spanish word or its meaning. |
| `audio-m` | Mean human iconicity rating for the auditory presentation subsample (Spanish). |
| `audio-sd` | Standard deviation of the auditory iconicity ratings (Spanish). |
| `audio-n` | Number of participants who rated auditory iconicity (Spanish). |
| `audio-dn` | Number of participants who did not know the word or meaning in the auditory condition (Spanish). |
| `gcat` | Grammatical category of the Spanish word (e.g., NOUN, INTERJECTION, ADJECTIVE). |
| `iconicity_spanish_base_prompt_gpt-4o` | Model-generated Spanish iconicity estimate from the base (non-fine-tuned) configuration; configuration lists `gpt-4o-mini-2024-07-18`. |
| `iconicity_spanish_base_prompt_gpt-4o_finetuned` | Model-generated Spanish iconicity estimate from the fine-tuned configuration defined in `config.yaml`. |

**Missing values:** auditory sub-study columns can be unavailable for words not included in the auditory sample; `#N/D` values should not be interpreted as numeric scores.
