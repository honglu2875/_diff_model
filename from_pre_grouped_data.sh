#!/bin/bash

python3 run_clm_diff.py --model_name_or_path=codegen-2b --per_device_train_batch_size=2 --ignore_long_samples --train_diff_model --save_final_dataset --concatenate_texts --num_train_epochs 1 --preprocessing_num_workers 60 --save_strategy=epoch --output_dir=diff_2b_full --report_to "wandb" --dataset_name pre_grouped_data --load_from_disk --tokenizer_name codegen-2b --block_size 2048 --gradient_accumulation_steps 2 --do_train --logging_strategy=epoch --overwrite_output_dir --adam_beta1=0.9 --adam_beta2=0.95 --weight_decay=2e-02 --learning_rate=1e-05 --warmup_steps=895 --per_device_eval_batch_size=1 --cache_dir="hf_cache_multinode" --gradient_checkpointing=True --force_label

