import os
import argparse
import lm_eval
import torch
import transformers
import bitsandbytes as bnb
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from src.llamafactory.rotation import fuse_layer_norms, fuse_layer_norms_noreplace, get_loaders, gptq_fwrd, rtn_fwrd, add_actquant, find_qlayers, get_hadK
from src.llamafactory.rotation import ActQuantWrapper, llama_down_proj_groupsize, rotate_model_global_only, rotate_model_r1r4, rotate_model_r4

DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
supported_datasets = ['wikitext2', 'ptb', 'c4']

def parser_gen():
    parser = argparse.ArgumentParser()
    # General Arguments
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',help='Model to load;')
    # LM Eval Arguments
    parser.add_argument('--tasks', type=str, default="boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa")
    parser.add_argument('--lm_eval_batch_size', type=int, default=1, help='Batch size for evaluating with lm eval harness.')
    args = parser.parse_args()

    return args

def replace_linear_with_4bit_linear(model):
    # Collect the modules to be replaced
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            modules_to_replace.append((name, module))

    # Perform the replacement
    for name, module in modules_to_replace:
        quantized_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None
        )
        quantized_linear.weight.data = module.weight.data
        if module.bias is not None:
            quantized_linear.bias.data = module.bias.data

        # Replace the original Linear layer
        parent_module = model
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], quantized_linear)
    
    return model

def main():
    args = parser_gen()
    print(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = transformers.LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model = replace_linear_with_4bit_linear(model)
    model.cuda()
    print("Evaluating Model: {}".format(model))
    model.seqlen = 2048 # assert we are using llama
    
    print("Finish Loading Model:\n{}".format(model))
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = args.tasks.split(',')
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)
    table_result = make_table(results)
    print(table_result)

    results = results['results']

    if args.tasks == 'gsm8k':
        print(results)
    elif args.tasks == 'mmlu':
        metric_vals = {}
        for task, result in results.items():
            if task in ['mmlu', 'mmlu_humanities', 'mmlu_other', 'mmlu_social_sciences', 'mmlu_stem']:
                metric_vals[task] = round(result.get('acc_norm,none', result['acc,none']), 4)
        print(metric_vals)
    else:
        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        print(metric_vals)

if __name__ == "__main__":
    main()