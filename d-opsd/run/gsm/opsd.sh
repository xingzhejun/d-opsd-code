#!/bin/bash
cd d-opsd
export WANDB_PROJECT="opsd_gsm8k"
export LOGDIR="checkpoints"

mkdir -p $LOGDIR

DATASET="gsm8k"
RUN_NAME=opsd
MODEL_PATH='GSAI-ML/LLaDA-8B-Instruct' 
PASSK=8 
PASSK_TEMP=0.9 
TEACHER_RETAIN_RATIO=0.25
BATCH_DIVIDE=4 # for A100 / H100, set to 8
# num_iter=barch_divide
TOP_K_LOSS=20
BETA=1
# debug1=True
fixed_teacher=True
add_ref=False
diff_student_mask=false
JSD_TOKEN_CLIP=0.05
if [ "$debug1" = "True" ]; then
    DEBUG_FLAG="--debug1"
else
    DEBUG_FLAG=""
fi
if [ "$fixed_teacher" = "True" ]; then
    FIXED_TEACHER_FLAG="--fixed_teacher"
else
    FIXED_TEACHER_FLAG=""
fi
if [ "$add_ref" = "True" ]; then
    ADD_REF_FLAG="--add_ref"
else
    ADD_REF_FLAG=""
fi
if [ "$diff_student_mask" = "True" ]; then
    DIFF_STUDENT_MASK_FLAG="--diff_student_mask"
else
    DIFF_STUDENT_MASK_FLAG=""
fi


accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12356 d_opsd_train.py \
    --config opsd.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $BATCH_DIVIDE \
    --batch_divide $BATCH_DIVIDE \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$DATASET/$RUN_NAME \
    --passk $PASSK \
    --passk_temperature $PASSK_TEMP \
    --teacher_retain_ratio $TEACHER_RETAIN_RATIO \
    --top_k_loss $TOP_K_LOSS \
    --beta $BETA \
    --jsd_token_clip $JSD_TOKEN_CLIP \
    $DEBUG_FLAG \
    $DIFF_STUDENT_MASK_FLAG \
    $ADD_REF_FLAG \
    $FIXED_TEACHER_FLAG