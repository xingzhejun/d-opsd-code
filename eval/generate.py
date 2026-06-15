import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist

from utils import main_print

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)

def calculate_distance_to_closest_unmasked(x_j, select_indices, mask_id):
    """
    For each index in select_indices, calculate the sum of distances to the
    1st and 2nd closest unmasked tokens in x_j.
    
    Args:
        x_j: 1D tensor of tokens for a single sample
        select_indices: tensor of indices to process
        mask_id: the mask token ID
    
    Returns:
        distances: tensor of distance sums for each selected index
    """
    # Find all unmasked token positions
    unmasked_mask = x_j != mask_id
    unmasked_indices = torch.nonzero(unmasked_mask, as_tuple=True)[0]
    
    distances = []
    
    for idx in select_indices:
        if len(unmasked_indices) < 2:
            # Not enough unmasked tokens
            distances.append(torch.tensor(0.0, device=x_j.device))
            continue
        
        # Calculate distances from idx to all unmasked indices
        dist_to_unmasked = torch.abs(unmasked_indices.float() - idx.float())
        
        # Get the 2 smallest distances
        top_2_distances = torch.topk(dist_to_unmasked, k=min(2, len(dist_to_unmasked)), largest=False)
        
        # Sum the distances
        distance_sum = top_2_distances.values.sum()
        distances.append(distance_sum)
    
    return torch.stack(distances) if distances else torch.tensor([], device=x_j.device)

@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_token_id=126081,
    calculate_distance=False,
    calculate_confidence=False,
    pre_answer=None,
    pre_answer_keep_mode="prefix",
    clean_answer=False,
    debug=False,
):
    """
    Optimized version of the generate function.
    """
    has_pre_answer = pre_answer is not None
    batch_size = prompt.shape[0]
    if pre_answer:
        pre_answer_tokens = tokenizer(pre_answer, return_tensors="pt").to(prompt.device)["input_ids"]
        pre_answer = pre_answer.replace("<|endoftext|>", "")
        pure_pre_answer_tokens = tokenizer(pre_answer, return_tensors="pt").to(prompt.device)["input_ids"]
        if pure_pre_answer_tokens.shape[1] > gen_length:
            pure_pre_answer_tokens = pure_pre_answer_tokens[:, -gen_length:]
            main_print(f"Warning: pre_answer length {pure_pre_answer_tokens.shape[1]} is longer than gen_length, truncating to {gen_length} tokens.")
        if debug:
            main_print(f'pre_token shape is: {pre_answer_tokens.shape}') # torch.Size([1, gen_length])
            main_print(f'pure_pre_token shape is: {pure_pre_answer_tokens.shape}') # torch.Size([1, effective_gen_length])

    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        prompt_index[:, : prompt.shape[1]] = True

        if has_pre_answer:
            # put pre_answer trajectories into each block with a ratio.
            # pre_answer_keep_mode: "prefix" | "random"
            known_answer = pure_pre_answer_tokens

            if known_answer.shape[0] == batch_size:
                if debug:
                    main_print(f'bsz=1')
                known_answer_batch = known_answer
            elif known_answer.shape[0] == 1:
                known_answer_batch = known_answer.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"pre_answer batch size ({known_answer.shape[0]}) is incompatible with prompt batch size ({batch_size})"
                )
            
            start_idx = prompt.shape[1]
            block_source = known_answer_batch
            valid_tokens = block_source.shape[1]
            sample_tokens = (valid_tokens // 4) # here is retaining 25%. You can adjust to 10% ot 50%, as shown in Table3 in the paper.
            if debug:
                main_print(f'valid_tokens is: {valid_tokens}, sample_tokens is: {sample_tokens}')
            if pre_answer_keep_mode == "prefix":
                x[:, start_idx : start_idx + sample_tokens] = block_source[:, :sample_tokens]
            elif pre_answer_keep_mode == "random":
                for b in range(batch_size):
                    rand_positions = torch.randperm(valid_tokens, device=prompt.device)[:sample_tokens]
                    x[b, start_idx + rand_positions] = block_source[b, rand_positions]
            else:
                raise ValueError(f"Unsupported pre_answer_keep_mode: {pre_answer_keep_mode}")

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        
        distances_list = []
        confidence_list = []
        stop_recording = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
        
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            if has_pre_answer and clean_answer:
                # prevent answer leaking
                x[:, start_idx:end_idx] = torch.full(
                    (batch_size, block_length), mask_id, dtype=torch.long, device=prompt.device
                    )
            
            block_mask_index = x[:, start_idx:end_idx] == mask_id
            if has_pre_answer:
                tokens_to_decode = int(block_mask_index[0].sum().item())
                steps_per_block = tokens_to_decode // 2
                if tokens_to_decode % 2 != 0:
                    steps_per_block += 1
                if steps_per_block == 0:
                    continue
            else:
                tokens_to_decode = block_mask_index.shape[1]
                steps_per_block = max(1, steps // num_blocks)

            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            if debug:
                main_print(f'token_to_decode for block {num_block} is: {tokens_to_decode}')
                main_print(f'steps_per_block for block {num_block} is: {steps_per_block}')
                main_print(f'num_transfer_tokens for block {num_block} is: {num_transfer_tokens}')
            # main_print(f'steps_per_block for block {num_block} is: {steps_per_block}')
            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Save confidence values for first step of each block
                if calculate_confidence and i == 0:
                    block_confidence = x0_p[:, start_idx:end_idx]
                    for j in range(block_confidence.shape[0]):
                        if stop_recording[j]:
                            continue
                        confidence_list.append({
                            'prompt_idx': j,
                            'block': num_block,
                            'step': i,
                            'confidence': block_confidence[j].cpu().tolist()
                        })

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    if calculate_distance and stop_recording[j]:
                        continue
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        
                        if calculate_distance:
                            distance = calculate_distance_to_closest_unmasked(x[j], select_indices, mask_id)
                            distances_list.append({
                                'prompt_idx': j,
                                'block': num_block,
                                'step': i,
                                'distances': distance.cpu().tolist()
                            })
                        
                        x[j, select_indices] = x0[j, select_indices]

            eos_present = torch.any(x[:, prompt.shape[1]:] == eos_token_id, dim=1)
            stop_recording |= eos_present
            # main_print(f'stop_recording is: {stop_recording}')
        
        return x, distances_list, confidence_list