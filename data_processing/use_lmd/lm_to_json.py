import lm_dataformat as lmd
from absl import flags
from absl import app
import os
import multiprocessing as mp
from util import get_files
from datasets import Dataset

FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'The input dir.')
flags.DEFINE_string('output', None, 'The output dir.')
flags.DEFINE_string('filename', 'out.json', 'The output filename.')
flags.DEFINE_integer('threads', 1, 'Number of threads.')


def read_from_file(input_path):
    reader = lmd.Reader(input_path)
    lst = list(reader.stream_data())
    print(f"Reading {input_path} completed.")

    return lst


def main(argv):
    if FLAGS.output:
        os.makedirs(FLAGS.output, exist_ok=True)

    files = get_files(FLAGS.input)
    print(f"{len(files)} files.")
    with mp.Pool(processes=FLAGS.threads) as pool:
        results = list(pool.map(read_from_file, files))

    docs = []
    for doc in results:
        docs.extend(doc)

    dataset = Dataset.from_list([{'text': s} for s in docs])
    dataset.to_json(FLAGS.output + "/" + FLAGS.filename)

   
if __name__=='__main__':
    app.run(main)
