"""
This is an all-in-one script to process jsonl.zst and turn them into huggingface's datasets with labels masked up to <DFF>
"""
import lm_dataformat as lmd
from transformers import AutoTokenizer
from absl import flags, app
from util import unravel, get_files
from kmp import kmp
from pathlib import Path
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import ast
import os, sys
from datasets import Dataset, load_dataset
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'The input dir.', short_name='i')
flags.DEFINE_string('output', None, 'The output dir.', short_name='o')
flags.DEFINE_integer('threads', 1, 'The number of threads.', short_name='th')
flags.DEFINE_integer('max_token_length', 2048, 'The maximal number of token length (longer samples will be filtered).', short_name='M')
flags.DEFINE_string('model', 'gpt2', 'The model name/path.', short_name='m')
flags.DEFINE_integer('chunk_size', 500_000, 'The size of each chunk.', short_name='ch')
flags.DEFINE_integer('min_batch', 0, 'The minimal amount of batches to produce.')



def main(argv):
    # Read the files in the folder
    counter = 0
    reader = lmd.Reader(FLAGS.input)
    
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    dff_tokens = tokenizer("<DFF>")["input_ids"]
    
    initial_hash = None
    not_ended = True
    generator = reader.stream_data()

    while not_ended:  # Will break based on whether reading the data is finished.
        output_path = FLAGS.output + f"/batch_{counter}"
        results = []
        for i, s in enumerate(generator):
            if i == 0 and counter == 0:
                initial_hash = hash(s)
            elif initial_hash == hash(s) and counter >= FLAGS.min_batch:
                not_ended = False  # This will be the last chunk
                break
            if i == FLAGS.chunk_size:
                break
            results.append(s)
        
        if not results:
            break

        logger.info(f"Finished reading batch {counter}. Total number of samples: {len(results)}")

        raw_datasets = Dataset.from_dict({"text": results})
        logger.info(raw_datasets.__repr__())

        def tokenize_function(examples, dff_tokens):
            tokenizer.model_max_length = int(1e8)  # Long samples will be discarded, so ignore the warnings.
            output = tokenizer(examples["text"], truncation=True)
            
            result = {key: [] for key in list(output.keys()) + ["labels", "invalid", "without_dff"]}
            for i in range(len(output["input_ids"])):
                result["invalid"].append(len(output["input_ids"][i]) > FLAGS.max_token_length)  # Mark long samples (to be filtered)
                index = kmp(output["input_ids"][i], dff_tokens, first_appearance=True)
                if not index:
                    result["invalid"][-1] = True
                    result["without_dff"].append(True)  # Mark invalid data (reflecting data quality. To be counted)
                    result["labels"].append([-100])
                else:
                    result["without_dff"].append(False)
                    index = index[0]
                    result["labels"].append([-100] * index + output["input_ids"][i][index:FLAGS.max_token_length].copy())

                for key in output:
                    result[key].append(output[key][i][:FLAGS.max_token_length])
                
            return result

        tokenized_datasets = raw_datasets.map(
            lambda examples: tokenize_function(examples, dff_tokens),
            batched=True,
            num_proc=FLAGS.threads,
            remove_columns=["text"],
            desc="Tokenize, filter length, apply mask.",
        )

        logger.info(f"Total sample count: {len(tokenized_datasets)}")
        logger.info(f"Invalid data count (without <DFF>): {sum(tokenized_datasets['without_dff'])}")
        logger.info(f"Excluded data count (either > {FLAGS.max_token_length} or without <DFF>): {sum(tokenized_datasets['invalid'])}")

        dataset = tokenized_datasets.filter(
                lambda examples: examples['invalid'] == False,
                num_proc=FLAGS.threads,
                desc="Remove invalid records."
        )

        logger.info(f"Final dataset summary:")
        logger.info(dataset.__repr__())

        dataset.save_to_disk(output_path)
        counter += 1

if __name__ == '__main__':
    app.run(main)
