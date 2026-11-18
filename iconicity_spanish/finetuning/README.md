# Spanish iconicity — finetuning

This folder belongs to the iconicity estimation experiments. The files listed below are those present in this repository snapshot.

## Files

| File | Purpose |
|---|---|
| [`ft_iconicity_spanish_base_prompt_gpt-4o.jsonl`](ft_iconicity_spanish_base_prompt_gpt-4o.jsonl) | Prepared JSONL records used to request Spanish model fine-tuning. |
| [`test_split_50.0_randomstate_42.xlsx`](test_split_50.0_randomstate_42.xlsx) | Held-out test/validation half of the human-rated norms (50/50 split; seed 42). |
| [`train_split_50.0_randomstate_42.xlsx`](train_split_50.0_randomstate_42.xlsx) | Training half of the human-rated norms (50/50 split; seed 42). |

## Split method

These files reflect the 50/50 train/test split configured with `train_split_percentage: 0.5` and `random_state: 42`. The selected answer is `rating` (English) or `ico-m` (Spanish). The fine-tuning JSONL is derived from the training portion; the test portion is reserved for validation.

**TO CONFIRM:** the exact API fine-tuning job settings (epochs, batch size, learning-rate multiplier, and returned job ID) from the archived job metadata.
