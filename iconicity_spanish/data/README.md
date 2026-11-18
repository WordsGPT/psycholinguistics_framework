# Spanish iconicity — data

This folder belongs to the iconicity estimation experiments. The files listed below are those present in this repository snapshot.

## Files

| File | Purpose |
|---|---|
| [`iconicity_ratings_spanish.xlsx`](iconicity_ratings_spanish.xlsx) | Human-rated iconicity norms used to fine-tune and/or validate the model. |
| [`List-Spanish-words.xlsx`](List-Spanish-words.xlsx) | Large original input list of lexical entries used for generating the extended iconicity estimates. |

Input data are preserved from the preceding lexical/norming datasets so that the entries can be aligned with earlier studies. The training/test files are not independent new human judgments.


### Data dictionary — `List-Spanish-words.xlsx`

Rows (excluding header): **127,728**. Columns: **2**.

| Column | Meaning |
|---|---|
| `Word` | Original lexical entry or expression preserved from the source word list (including punctuation and numeric forms). |
| `Source` | Source label or lexical resource associated with the entry (e.g., RAE_dic, fundeu_dic, Dictionary_on_web, BPAL, published studies); use the cited source study for full provenance. |


### Data dictionary — `iconicity_ratings_spanish.xlsx`

Rows (excluding header): **10,995**. Columns: **10**.

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


## Source studies

- Extended word list: Martínez et al. (2025), AI-generated estimates of familiarity, concreteness, valence, and arousal for over 100,000 Spanish words; source data: https://osf.io/frc6a/files/9672j
- Human-rating norms: Hinojosa et al. (2021), Iconicity ratings for 10,995 Spanish words: https://doi.org/10.3758/s13428-020-01496-z


**Note on splits:** The corresponding files in `../finetuning/` use the same columns as `iconicity_ratings_spanish.xlsx`.
