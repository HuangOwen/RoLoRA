from .utils import cleanup_memory
from .hadamard_utils import random_hadamard_matrix, is_pow2, apply_exact_had_to_linear
from .model_utils import get_model_type, get_embeddings, get_transformer_layers, get_pre_head_layernorm, get_lm_head, replace_modules, model_type_extractor
from .model_utils import LLAMA_MODEL, OPT_MODEL, RMSN
from .utils import DEV

import torch
import typing
import transformers
import tqdm


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def reinitialize_rmsnorm(layer):
    if isinstance(layer, transformers.models.llama.modeling_llama.LlamaRMSNorm):
        with torch.no_grad():
            layer.weight.fill_(1.0)  # Reset the weights to 1.0 or other desired value

def fuse_layer_norms_noreplace(model):
    
    model_type = get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(get_pre_head_layernorm(**kwargs), [get_lm_head(**kwargs)])
    model.apply(reinitialize_rmsnorm)
         
def fuse_layer_norms(model):
    
    model_type = get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(get_pre_head_layernorm(**kwargs), [get_lm_head(**kwargs)])
    
    replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    
def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_type_extractor(model)
    for W in get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
def rotate_down_proj(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_output_nohad(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)

def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = get_lm_head(model, model_type=model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

@torch.inference_mode()
def rotate_model_r4(model, rotate_mode):
    print("Rotating R4")
    Q = get_orthogonal_matrix(model.config.hidden_size, rotate_mode)
    model_type = model_type_extractor(model)
    layers = get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_down_proj(layers[idx], Q, model_type)


@torch.inference_mode()
def rotate_model_r1r4(model, rotate_mode):
    print("Rotating R1 and R4")
    Q = get_orthogonal_matrix(model.config.hidden_size, rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    model_type = model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    cleanup_memory()
    layers = get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type)

@torch.inference_mode()
def rotate_model_global_only(model, rotate_mode):
    print("Rotating Shortcut Only! (No online hadamard)")
    Q = get_orthogonal_matrix(model.config.hidden_size, rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size

    model_type = model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    cleanup_memory()
    layers = get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output_nohad(layers[idx], Q, model_type)


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)
