import lm_dataformat as lmd
from util import get_files, unravel
from absl import flags
from absl import app
import ast

FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'The input dir.')
flags.DEFINE_string('index_dir', None, 'The index dir.')
flags.DEFINE_string('output', None, 'The output dir.')



def main(argv):
    inp_path = FLAGS.input
    out_path = FLAGS.output
    ind_path = FLAGS.index_dir

    files = get_files(FLAGS.input)

    for file in files:
        print(f"Reading file {file}")
        filename = file.split("/")[-1]
        for name, filepath in [("input", inp_path + "/" + filename), ("output", out_path + "/" + filename)]:
            reader = lmd.Reader(filepath)
            print("  " + name + " length: " + str(len(list(reader.stream_data()))))

        with open(ind_path + "/" + filename, 'r') as f:
            lst = unravel(ast.literal_eval(f.read()))
            print("  index length: " + str(len(lst)))


if __name__=="__main__":
    app.run(main)
