# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ...rotation import fuse_layer_norms, rotate_model_global_only
from ..utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

import torch
from scipy.stats import kurtosis
from transformers import TrainerCallback

# Define a function to compute kurtosis
def compute_kurtosis(tensor):
    # Flatten the tensor to a 1D array
    tensor_flat = tensor[0].float().view(-1)
    # Compute mean and standard deviation
    mean = torch.mean(tensor_flat)
    std = torch.std(tensor_flat)
    # Compute kurtosis using the PyTorch functions
    n = tensor_flat.numel()
    normalized_tensor = (tensor_flat - mean) / std
    kurtosis = torch.mean(normalized_tensor ** 4) - 3

    return kurtosis.item()

def save_activations_hook(module, input, output, activations_dict, layer_name):
    #if len(input) > 0 and len(output) > 0 and 'base_layer' in layer_name:
        #input_kurtosis = compute_kurtosis(input)
        #output_kurtosis = compute_kurtosis(output)
        #print("Kurtosis for {}: Input {} Output {}".format(layer_name, input_kurtosis, output_kurtosis))
    activations_dict[layer_name] = {
        'input': input,
        'output': output
    }

def register_hooks(model, activations):
    for name, module in model.named_modules():
        if 'base_layer' in name:
            module.register_forward_hook(
                lambda module, input, output, name=name: save_activations_hook(module, input, output, activations, name)
            )

class SaveActivationsCallback(TrainerCallback):
    def __init__(self, activations, save_steps):
        self.activations = activations
        self.save_steps = save_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            # Save activations to disk or process them as needed
            torch.save(self.activations, f'/project/vislangmod/xijie/LLaMA-Factory/tmp_activation/baseline_lora_full_activation_part/activations_step_{state.global_step}.pt')
            # Clear activations dictionary to save memory
            self.activations.clear()

# Example usage:
# activations = {}
# save_steps = [10, 20, 30]  # Specify the steps at which you want to save activations
# save_activations_callback = SaveActivationsCallback(activations, save_steps)

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # register hook for activation saving
    # activations = {}
    # register_hooks(model, activations)
    # save_steps = [500*i for i in range(0,35)]  # standard lora training
    # save_activations_callback = SaveActivationsCallback(activations, save_steps)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        #callbacks=[save_activations_callback],
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
