# English iconicity — data

This folder belongs to the iconicity estimation experiments. The files listed below are those present in this repository snapshot.

## Files

| File | Purpose |
|---|---|
| [`iconicity_ratings_english.xlsx`](iconicity_ratings_english.xlsx) | Human-rated iconicity norms used to fine-tune and/or validate the model. |
| [`iconicity_ratings_english_sample.xlsx`](iconicity_ratings_english_sample.xlsx) | Small human-rated sample for quick development/testing; not the complete input dataset. |
| [`iconicity_ratings_english_sample_2.xlsx`](iconicity_ratings_english_sample_2.xlsx) | Small human-rated sample for quick development/testing; not the complete input dataset. |
| [`List_English-Words.xlsx`](List_English-Words.xlsx) | Large original input list of lexical entries used for generating the extended iconicity estimates. |
| [`test_split_50.0_randomstate_42.xlsx`](test_split_50.0_randomstate_42.xlsx) | Held-out test/validation half of the human-rated norms (50/50 split; seed 42). This path is a symlink to the corresponding file under `finetuning/`. |
| [`train_split_50.0_randomstate_42.xlsx`](train_split_50.0_randomstate_42.xlsx) | Training half of the human-rated norms (50/50 split; seed 42). This path is a symlink to the corresponding file under `finetuning/`. |

Input data are preserved from the preceding lexical/norming datasets so that the entries can be aligned with earlier studies. The training/test files are not independent new human judgments.


### Data dictionary — `List_English-Words.xlsx`

Rows (excluding header): **65,305**. Columns: **1**.

| Column | Meaning |
|---|---|
| `Word` | Original lexical entry or expression preserved from the source word list (including punctuation and numeric forms). |


### Data dictionary — `iconicity_ratings_english.xlsx`

Rows (excluding header): **14,776**. Columns: **6**.

| Column | Meaning |
|---|---|
| `word` | Lexical entry or expression as represented in this table; case and punctuation may be meaningful. |
| `n_ratings` | Number of ratings recorded in the English source dataset. **CHECK:** exact counting rule versus `n` should be confirmed against the original dataset documentation. |
| `n` | Count field retained from the English source norms. **CHECK:** verify the exact denominator and the distinction from `n_ratings` with Winter et al. (2024). |
| `prop_known` | Proportion of participants indicating that they knew the word; inherited from the English norms (usually between 0 and 1). |
| `rating` | Mean human iconicity rating on a 1–7 scale (English norms; Winter et al., 2024). |
| `rating_sd` | Standard deviation of the human iconicity ratings (English normative study). |


## Source studies

- Extended word list: Brysbaert, Martínez & Reviriego (2025), Moving beyond word frequency based on tally counting; source data: https://osf.io/c2yef/files/bafpv
- Human-rating norms: Winter et al. (2024), Iconicity ratings for 14,000+ English words: https://doi.org/10.3758/s13428-023-02112-6


**Note on splits:** `train_split_50.0_randomstate_42.xlsx` and `test_split_50.0_randomstate_42.xlsx` have the same fields as `iconicity_ratings_english.xlsx`. The paths here are symbolic links to the files in `../finetuning/`.
