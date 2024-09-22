import os
import utils
import torch
import model_utils
import data_utils
import transformers
import rotation_utils
import eval_utils
import argparse
import pprint
import logging

supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
# General Arguments
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',help='Model to load')
parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                    help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
parser.add_argument('--hf_token', type=str, default=None)
parser.add_argument('--bsz', type=int, default=32,
                    help='Batch-size for PPL evaluation (default:32)')

# Rotation Arguments
parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                    help='''Rotate the moodel. This will include online rotation for down-projection and
                    out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                    if we want to quantize the Keys''')
parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
parser.add_argument('--rotation_seed', type=int, default=-1,
                    help='Random Seed for generating random matrix!!')

################################ Set args ################################
args = parser.parse_args(['--model', 'meta-llama/Llama-2-7b-hf', '--bsz', '1', '--rotate', '--rotate_mode', 'hadamard'])
################################ Set args ################################
  
logging.info('Arguments: ')
logging.info(pprint.pformat(vars(args)))
logging.info('--' * 30)

transformers.set_seed(args.seed)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto',device_map="auto",low_cpu_mem_usage=True)
model.seqlen = 2048
logging.info('---> Loading {} Model with seq_len: {}'.format(args.model, model.seqlen))

rotation_utils.fuse_layer_norms(model)
rotation_utils.rotate_model_global_only(model, args.rotate_mode)

# Evaluating on dataset
testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True
    )

dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
print("PPL on {} after rotataion: {}".format(args.eval_dataset, dataset_ppl))