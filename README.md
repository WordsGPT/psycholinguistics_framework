# Psycholinguistics framework

## Setting Up the framework
1. Copy `apis_example.env` to `apis.env` and fill it with your model credentials
2. Install dependencies with `pip install -r requirements.txt`

## Setting Up a New Experiment

### Initial configuration
1. Create a folder with the name of the experiment, add the data in XLSX format in the `data` subfolder, and possible prompts in TXT format in the `prompts` subfolder.
2. Create a `config.yaml` file in the experiment folder using the content of `config_example.yaml` as a template.

### Make estimations with some model (including fine-tuning models):
1. Prepare the experiment by running `python3 prepare_experiment.py <EXPERIMENT_PATH> <EXPERIMENT NAME>`. -> generates the batches
2. Save the batches.jsonl files in `batches` folder
3. Run the experiment by executing `python3 execute_experiment.py <EXPERIMENT_PATH>`. ->  executes the batches.

### Processing results:
1. Save the results of batches in `results` folder
2. Combine all the batches in a file called `batches.jsonl`
3. Combine all the results files in a file called `results.jsonl`

You can use these commands:
```
cat batches/*.jsonl >> batches/batches.jsonl
cat results/*.jsonl >> results/results.jsonl
```
4. Execute `python3 generateResults.py [extra-otpion]` for the experiment. Each experiment will have its own `generateResults.py` -> it generates a .xlsx with the results

### Make a finatuning:
1. Prepare the fine-tuning dataset by running `python3 create_finetuning_dataset.py <EXPERIMENT_PATH> <EXPERIMENT NAME>`.
2. Fine-tune the model by running `python3 execute_finetune.py <EXPERIMENT_PATH> <FT_NAME>`.
3. Calculate the correlation by running `python3 calculate_correlation.py <EXPERIMENT_PATH>`.
4. To test the finetuning execute the same steps mentioned before.

### Other scripts:
- `execute_individual_api.py`: Instead of batch operations executes individual calls to the model API
- `combine_excels.py`: It makes a left join of file1.xlsx and file2.xlsx using two columns and generates a new excel with the same name as file1.xlsx (saving the initial one as file1_old.xlsx) `python3 combine_excels.py <file1.xlsx> <file2.xlsx> <Column-file1,Column-file2> <suffix for columns of file2.xlsx>`

