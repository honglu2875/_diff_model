# _diff_model
documenting scripts and workflows for diff model training.

**This is not intended for easy-usage. It is merely a record.**

# General workflows
- Prepare the original dataset for huggingface `load_dataset` to read (e.g. a json file. For jsonl.zst file, it would be fine too but would need a custom loading script).
  - One can also start directly from hf datasets saved by `.save_to_disk` method. My script `run_clm_diff.py` will use `load_from_disk` api when seeing `--load_from_disk` flag (with folder name specified in `--dataset_name`).
- Prepare the running script (be it an sbatch script or just a bash script). Look for the following new flags I introduced (detailed explanation see the top of source code):
  - `--concatenate_texts`: Only pass it when we want to try out the idea of concatenating all the texts and truncate into equal-size chunks (separated by EOS token). If not passed, we pad every string to the left by EOS.
  - `--train_diff_model`: Always pass it for diff models.
  - `--ignore_long_samples`: Throw away samples of size larger than or **equal** to the block_size (leave one space for EOS). If not, we automatically force `--concatenate_texts` because padding doesn't make sense.
  - `--load_from_disk`: Load the data using `load_from_disk` instead.
  - `--save_final_dataset`: After all the preprocessing (load->tokenize->mask(before `<DFF>`)->concat or pad), save the final dataset into `final_dataset` folder using `.save_to_disk` method.
  - `--skip_concat`: Skip the last step of data preprocessing (concatenate or pad).
  - `--force_label`: Force the third step (generate `labels` by masking tokens before `<DFF>`) even if we already have `labels` column in the data.
- Run on the cluster

Practically, we may need to break down the data preprocessing part several times (thus so many flags to load from mid-points). In `run_clm_diff.py` script, there is a data sanity check before training. But please also do some basic manual inspections to make sure there is no unexpected data processing behavior.

# How `run_clm_diff.py` works now
I have added a few things to this finetuning script. Now, it does the following things in order:

1. load dataset (containing at least 'train' split).
2. add in dropout rate to the loaded model.
3. tokenize (skip if `input_ids` is a column). Possibly throw away long samples (`--ignore_long_samples`).
4. generate `labels` field by masking tokens before `<DFF>` (unless `--force_label` is **not** set and `labels` is already a column).
5. either concatenate or pad: **concatenate**: add an EOS to the end of every sample, and then concatenate all of them before truncating to `block_size`; **pad**: add EOS to the left of every sample. Fill `labels` with `-100` and `attention_mask` with `0` accordingly.
6. sanity check, possibly save final dataset (`--save_final_dataset`).
7. pass into `Trainer`.

