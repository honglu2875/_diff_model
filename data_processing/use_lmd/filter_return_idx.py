"""
This will tokenize all entries in jsonl.zst files, check the token length, 
and return the label number of those that are less than the given number (max_token).

It will generate a text file under the path specified by `output` (is a directory).
"""



from transformers import AutoTokenizer
import tensorflow as tf
import lm_dataformat as lmd
from absl import flags
from absl import app
import os
from pathlib import Path
from random import random
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'The input folder for jsonl.zst.')
flags.DEFINE_string('output', None, 'The output directory.')
flags.DEFINE_string('tokenizer_path', 'gpt2', 'tokenizer path.')
flags.DEFINE_integer('threads', 1, 'number of threads.')
flags.DEFINE_integer('max_token', 2048, 'throw away samples with more token numbers.')
flags.DEFINE_string('name', 'data', 'file prefix.')
flags.DEFINE_bool('separate_by_file', False, 'separately process each file by different processes.')
flags.DEFINE_integer('max_sample_per_call', 100_000, 'maximal number of samples to tokenize in each tokenizer call.')



def tokenize(thread_id, lst, tokenizer, offset):
    NUM = FLAGS.max_sample_per_call
    LEN = (len(lst)-1) // NUM + 1

    ids = []
    for i in range(LEN):
        chunk = lst[i*NUM: (i + 1) * NUM]
        result = tokenizer(chunk, return_tensors='np', max_length=int(1e8), truncation=True)['input_ids']
        for j in range(len(result)):
            if len(result[j]) < FLAGS.max_token:
                ids.append(j + offset)

        del result
        logger.info(f"Worker {thread_id}: Chunk {i} finished.")
        return ids


def filter_token_len(docs, file_path, num_workers=None, offset=0):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_path)
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    NUM_PROCESS = FLAGS.threads if num_workers is None else num_workers
    LEN = len(docs) // NUM_PROCESS

    parallel_args = [(i, docs[i*LEN:(i+1)*LEN], tokenizer, i*LEN + offset) for i in range(NUM_PROCESS)]
    if NUM_PROCESS == 1:
        result = [tokenize(0, docs, tokenizer, 0)]
    else:
        with mp.Pool(processes=NUM_PROCESS) as pool:
            result = list(pool.starmap(tokenize, parallel_args))
        logger.info("All processes finished.")

    
    logger.info(f"Writing to text file at {file_path}")

    with open(file_path, "w") as f:
        f.write(str(result))


def get_files(input_path):
    supported_file_types = ['jsonl.zst']
    
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if input_path.is_dir():
        # get all files with supported file types
        files = [list(Path(input_path).glob(f"*{ft}")) for ft in supported_file_types]
        # flatten list
        files = [f for sublist in files for f in sublist]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert any(
            str(input_path).endswith(f_type) for f_type in supported_file_types
        ), f"Input file type must be one of: {supported_file_types}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path}")

    return [str(f) for f in files]


def read_from_file(input_path):

    file_path = FLAGS.output + "/" + input_path.split("/")[-1] 
    
    if FLAGS.separate_by_file:
        if not Path(file_path).exists():
            reader = lmd.Reader(input_path)
            lst = list(reader.stream_data())
            logger.info(f"Reading {input_path} completed.")
            filter_token_len(lst, file_path=file_path, num_workers=1)
        else:
            logger.info(f"{file_path} already exists.")
        
        return None  # Let garbage collector take everything. Save memory footprint.
    else:
        reader = lmd.Reader(input_path)
        lst = list(reader.stream_data())
        logger.info(f"Reading {input_path} completed.")
        
        return lst


def main(argv):
    if FLAGS.output:
        os.makedirs(FLAGS.output, exist_ok=True)
    
    logger.info("Started.")
    
    # Read the files in the folder
    files = get_files(FLAGS.input)
    with mp.Pool(processes=FLAGS.threads) as pool:
        results = pool.map(read_from_file, files)
    
    if not FLAGS.separate_by_file:
        # Flatten the list
        docs = []
        for doc in results:
            docs.extend(doc)
        logger.info(f"Reading finished. Tokenizing.")

        file_path = FLAGS.output + f"/{FLAGS.name}.txt"
        filter_token_len(docs, file_path=file_path)

if __name__=="__main__":
    app.run(main)

