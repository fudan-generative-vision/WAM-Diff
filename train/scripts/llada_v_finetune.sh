#!/bin/bash
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_TIMEOUT=22  # 平台要求from马国权 

num_node=${WORLD_SIZE:-1}
gpu_num=${NGPUS_PER_NODE:-1}

save_dir=${BASE_DIR:-'exp'}
BASE_RUN_NAME=${BASE_RUN_NAME:-"moe_lora_s2"}
total_epoch=${EPOCH:-3}
saving_freq=${SAVE_FREQ:-5000}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

pretrain_path=${PRETRAIN_MODEL:-"../model/WAM-Diff-ckpt"}
LLM_VERSION=${pretrain_path}
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="../model/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
data_path=${DATA_PATH:-"./data/pretrain_moe.json"}

############### Finetune ################

PROMPT_VERSION="llava_llada"

echo "save dir: ${save_dir}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "total epoch: ${total_epoch}"
echo "saving freq: ${saving_freq}"
echo "pretrained model path: ${pretrain_path}"
echo "data path: ${data_path}"

echo "
moe_choice: ${MOE_CHOICE:-'expert'}
num_experts_per_tok: ${NUM_EXPERT_PER_TOK:-'64'}
num_experts: ${NUM_EXPERT:-'64'}
capacity_factor: ${CAPACITY_FACTOR:-'0.1'}
moe_router_score_function: None
moe_lora_rank: ${MOE_LORA_RANK:-32}
moe_lora_alpha: ${MOE_LORA_ALPHA:-32}
moe_lora_in_features: 4096
moe_lora_out_features: 4096
moe_lora_dropout: 0.05
"
export TRAIN_MODULE=${TRAIN_MODULE:-"mm_vision_tower,mm_mlp_adapter,mm_language_model"}
echo $TRAIN_MODULE

export PYTHONPATH=$PYTHONPATH:${PWD}
echo $PYTHONPATH

export ZERO_STAGE_CONFIG=${ZERO_STAGE_CONFIG:-"scripts/zero3.json"}

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed ${ZERO_STAGE_CONFIG} \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "$data_path" \
    --mm_tunable_parts ${TRAIN_MODULE} \
    --num_token ${NUM_TOKEN:-False} \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_2 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --moe_choice ${MOE_CHOICE:-'expert'} \
    --num_experts_per_tok ${NUM_EXPERT_PER_TOK:-'64'} \
    --num_experts ${NUM_EXPERT:-'64'} \
    --moe_router_score_function None \
    --moe_lora_rank ${MOE_LORA_RANK:-32} \
    --moe_lora_alpha ${MOE_LORA_ALPHA:-32} \
    --moe_lora_in_features 4096 \
    --moe_lora_out_features 4096 \
    --moe_lora_dropout 0.05 \
    --capacity_factor ${CAPACITY_FACTOR:-'0.1'} \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "$save_dir/$BASE_RUN_NAME" \
    --num_train_epochs ${total_epoch} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps ${saving_freq} \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --fp16 False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --debug_mode False