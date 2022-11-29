from transformers import AutoTokenizer, AutoModelForCausalLM
from absl import flags, app
import torch


sample = """<NME> test.py
<BEF> def add(x, y):
    z = x + y
    print(f'result: {z}')

if __name__=='__main__':
    add(5, 6)
<MSG> change add to multiply
"""

FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, 'Pick your poison: finetuned_codegen_350m, finetuned_codegen_2b, finetuned_codegen_6b.', short_name='m')
flags.DEFINE_string('input_str', sample, 'The input prompt.', short_name='i')
flags.DEFINE_integer('max_length', 200, 'The max length to generate.')
flags.DEFINE_integer('num_beams', 1, 'Num beams (1 if greedy).')
flags.mark_flag_as_required('model')


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    model = AutoModelForCausalLM.from_pretrained(FLAGS.model).to(device)

    tokens = tokenizer(FLAGS.input_str, return_tensors='pt').to(device)
    result = model.generate(**tokens, max_length=FLAGS.max_length, num_beams=FLAGS.num_beams)
    print(tokenizer.batch_decode(result)[0])

if __name__ == '__main__':
    app.run(main)
