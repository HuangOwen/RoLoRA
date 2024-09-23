NAME="llama2-7b-rolora-qv-r16"
PRETRAINED="meta-llama/Llama-2-7b-hf"
WBITS=4
ABITS=4

CUDA_VISIBLE_DEVICES=0 python llm_eval.py \
--model ./models/$NAME/ \
--pretrained_model $PRETRAINED \
--lm_eval_batch_size 16 \
--rotate_down_proj \
--task boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
CUDA_VISIBLE_DEVICES=0 python llm_eval.py \
--model ./models/$NAME/ \
--pretrained_model $PRETRAINED \
--lm_eval_batch_size 16 \
--rotate_down_proj \
--w_rtn \
--w_bits $WBITS \
--a_bits $ABITS \
--w_clip \
--task boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
CUDA_VISIBLE_DEVICES=0 python llm_eval.py \
--model ./models/$NAME/ \
--pretrained_model $PRETRAINED \
--lm_eval_batch_size 16 \
--rotate_down_proj \
--w_bits $WBITS \
--a_bits $ABITS \
--w_clip \
--task boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

CUDA_VISIBLE_DEVICES=0 python llm_eval.py \
--model ./models/$NAME/ \
--pretrained_model $PRETRAINED \
--lm_eval_batch_size 16 \
--rotate_down_proj \
--task mmlu
CUDA_VISIBLE_DEVICES=0 python llm_eval.py \
--model ./models/$NAME/ \
--pretrained_model $PRETRAINED \
--lm_eval_batch_size 16 \
--rotate_down_proj \
--w_rtn \
--w_bits $WBITS \
--a_bits $ABITS \
--w_clip \
--task mmlu
CUDA_VISIBLE_DEVICES=0 python llm_eval.py \
--model ./models/$NAME/ \
--pretrained_model $PRETRAINED \
--lm_eval_batch_size 16 \
--rotate_down_proj \
--w_bits $WBITS \
--a_bits $ABITS \
--w_clip \
--task mmlu
