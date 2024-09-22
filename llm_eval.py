import os
import argparse
import lm_eval
import torch
import transformers
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
    parser.add_argument('--pretrained_model', type=str, default='/project/vislangmod/Llama-2-7b-hf/',help='Pretrained Model (to build architecture);')
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                        help='Whether to perform rotation on the model')
    parser.add_argument('--rotate_down_proj', action=argparse.BooleanOptionalAction, default=False, 
                        help='Whether to perform rotation on the down_proj in LLaMA MLP')
    parser.add_argument('--rotate_from_pretrained', action=argparse.BooleanOptionalAction, default=False, 
                        help='Whether to perform rotation from the pretrained model')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False, help='Apply Hadamard rotation in FP32 (default: False)')
    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')
    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')

    # LM Eval Arguments
    parser.add_argument('--tasks', type=str, default="boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa")
    parser.add_argument('--lm_eval_batch_size', type=int, default=1, help='Batch size for evaluating with lm eval harness.')
    args = parser.parse_args()

    return args

def main():
    args = parser_gen()
    print(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if args.rotate_down_proj:
        empty_model = transformers.LlamaForCausalLM.from_pretrained(args.pretrained_model, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
        fuse_layer_norms_noreplace(empty_model)
        add_actquant(empty_model) 
        qlayers = find_qlayers(empty_model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = get_hadK(empty_model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
        rotate_model_state_dict = torch.load(os.path.join(args.model, 'rotate_model.bin'))
        result = empty_model.load_state_dict(rotate_model_state_dict, strict=True)
        print("Loading Roatated Model Results: {}".format(result))
        del rotate_model_state_dict
        torch.cuda.empty_cache() 
        model = empty_model
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
        if 'LoftQ' in args.model:
            print("Dequantizing LoftQ Models")
            model.dequantize()
            model.quantization_method = None

    print("Evaluating Model: {}".format(model))
    model.seqlen = 2048 # assert we are using llama

    if args.w_bits < 16:
        # Add Activation Wrapper to the model as the rest of the code assumes it is present
        if not args.rotate_down_proj:
            add_actquant(model)
        if not args.w_rtn: # GPTQ Weight Quantization
            trainloader = get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_fwrd(model, trainloader, DEV, args)
        else: # RTN Weight Quantization
            quantizers = rtn_fwrd(model, DEV, args)

    if args.a_bits < 16:
        qlayers = find_qlayers(model, layers=[ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
                
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)
    
    if args.a_bits < 16 or args.w_bits < 16:
        model.eval()
        model.to(DEV)
    
    print("Finish Loading Model:\n{}".format(model))
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = args.tasks.split(',')
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size, device=DEV)
    table_result = make_table(results)
    print(table_result)

    if args.w_bits == 16 and args.a_bits == 16:
        file_name = "{}_FP16.log".format(args.tasks)
    elif args.w_rtn:
        file_name = "{}_W{}_A{}_RTN.log".format(args.tasks, args.w_bits, args.a_bits)
    elif args.rotate_from_pretrained:
        file_name = "{}_W{}_A{}_GPTQ-Rotate.log".format(args.tasks, args.w_bits, args.a_bits)
    else:
        file_name = "{}_W{}_A{}_GPTQ.log".format(args.tasks, args.w_bits, args.a_bits)
    file_path = os.path.join(args.model, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(table_result)

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