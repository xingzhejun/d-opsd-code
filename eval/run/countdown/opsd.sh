#!/bin/bash
cd eval

# Configuration variables
GPU_IDS=(0)
MASTER_PORT=18040

# Arrays of tasks and generation lengths
TASKS=("countdown")
GEN_LENGTHS=(128) # (128 256)
DIFFUSION_STEPS=(64) # (64, 128)
BLOCK_LENGTHS=(32)
temperature=0.0
num_answer_per_question=1
REMASKING=("low_confidence")
CHECKPOINT_PATH=("path/to/your/checkpoints") # TODO: Update this with the actual path to your checkpoints

# Set GPU IDs from command line if provided`
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    for diffusion_step in "${DIFFUSION_STEPS[@]}"; do
      for remasking in "${REMASKING[@]}"; do
        for block_length in "${BLOCK_LENGTHS[@]}"; do
            for chekpoint in "${CHECKPOINT_PATH[@]}"; do
                # Set batch size based on generation length
                if [ "$gen_length" -eq 512 ]; then
                batch_size=4
                else
                batch_size=8
                fi

                # Skip invalid combinations. We fix of decoding 2 tokens each step.
                if (( 2 * diffusion_step != gen_length )); then
                echo "Skipping combination: block_length=$block_length diffusion_step=$diffusion_step (product != gen_length=$gen_length)"
                continue
                fi
                echo "Running evaluation on $task with gen_length=$gen_length, diffusion_step=$diffusion_step, remasking=$remasking, block_length=$block_length, temperature=$temperature, num_answer_per_question=$num_answer_per_question"
                
                CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
                --nproc_per_node $NUM_GPUS \
                --master_port $MASTER_PORT \
                eval.py \
                --dataset $task \
                --batch_size $batch_size \
                --gen_length $gen_length \
                --output_dir "generations" \
                --model_path "GSAI-ML/LLaDA-8B-Instruct" \
                --checkpoint_path "$chekpoint" \
                --diffusion_steps $diffusion_step \
                --remasking "$remasking" \
                --block_length $block_length \
                --temperature $temperature \
                --seed 42 \
                --num_answer_per_question $num_answer_per_question \
                --split "test"
            done
        done
      done
    done
  done
done


echo "All evaluations completed!"


# --calculate_distance \
# --calculate_confidence \

### the following are for the toy experiment in Table3 in the paper.
# --answer_path "path/to/your/pre-answer" \
# --pre_answer_keep_mode "random" \
# --clean_answer \
# please also remember to update the split to "train"