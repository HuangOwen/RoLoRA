import os
import torch
import argparse
from safetensors.torch import load_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='path to generate torch model from safetensor files')
    args = parser.parse_args()
    state_dict = {}
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith(".safetensor"):
                loaded_state_dict = load_file(os.path.join(root, file))
                state_dict.update(loaded_state_dict)
    torch.save(state_dict, os.path.join(root, 'rotate_model.bin'))
    print("Finishing Mergeing Statedict!")
                






    
