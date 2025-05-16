# import copy
import logging

import numpy as np

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from model.pld.pld import find_candidate_pred_tokens

from .kv_cache import clone_past_key_values


def set_logger(log_path=None):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
            )
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> LogitsProcessorList:
    """
    Prepare the logits processor based on the provided parameters.

    Parameters:
    - temperature (float): Softmax temperature for probability scaling.
    - repetition_penalty (float): Penalty for repeating tokens.
    - top_p (float): Nucleus sampling probability threshold.
    - top_k (int): Top-k sampling threshold.

    Returns:
    - LogitsProcessorList: A list of processors to apply to the logits.
    """
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list

def sample(logits, logits_processor, all_logits=False, k=1):
    """
    Sample from the provided logits using the specified processor.

    Args:
    - logits (torch.Tensor): Logits to sample from.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.
    - k (int): Number of samples to generate.

    Returns:
    - sampled_indices (torch.Tensor): Indices of the sampled tokens.
    - sampled_probs (torch.Tensor): Probabilities of the sampled tokens.
    - probabilities (torch.Tensor): Probabilities of all tokens.
    """
    if logits_processor is None:  # greedy decoding
        # sample_token = torch.argmax(logits[:, -1])
        # # sample_token = torch.argmax(logits)
        # sample_token = sample_token[None, None]
        if all_logits:
            sample_indices = torch.argmax(logits, dim=-1)
        else:
            sample_indices = torch.argmax(logits[:, -1])
            sample_indices = sample_indices[None, None]
        return sample_indices, None, None

    logits = logits.view(-1, logits.size(-1))  # default batch size 1
    logits = logits_processor(None, logits)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    sampled_indices = torch.multinomial(probabilities, k, replacement=False)
    sampled_probs = torch.gather(probabilities, -1, sampled_indices)

    cumulative_sum = torch.cumsum(sampled_probs, dim=-1)
    cumulative_sum = torch.cat(
        (
            torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device),
            cumulative_sum[:, :-1],
        ),
        dim=-1,
    )

    sampled_probs = sampled_probs / (1 - cumulative_sum)
    sampled_probs[torch.isinf(sampled_probs)] = -1
    sampled_probs[torch.isnan(sampled_probs)] = -1

    sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

    return sampled_indices, sampled_probs, probabilities


@torch.no_grad()
def clasp_draft(
    model,
    device,
    input_ids_list=None,
    current_draft_skip_mask=None,
    new_token_num=0,
    past_key_values_data=None,
    current_length_data=None,
    max_new_tokens=1024,
    position_ids=None,
    K=25,
    DET=0.7,
    eos_token_id=None,
    logits_processor=None,
    HC = False,
    VC = False,
    tree = False,
):
    draft_tokens = []
    # Use a *cloned* KV cache structure for drafting
    # We need to clone the underlying data tensors and create new KVCache objects
    draft_past_key_values_data_list = [d.clone() for d in past_key_values_data]
    draft_current_length_data = (
        current_length_data.clone()
    )  # This tracks lengths for the draft cache
    # Create new KVCache objects pointing to the *cloned* data
    draft_kv_cache_list = clone_past_key_values(
        model, draft_past_key_values_data_list, draft_current_length_data
    )

    first_draft_input_ids = torch.tensor(
        [input_ids_list[0][-1]], device=device
    ).unsqueeze(0)
    next_draft_input_ids = first_draft_input_ids
    continue_hc = True
    for draft_step_idx in range(K):
        # logging.info(f"token count: {len(draft_tokens) + generated_token_count}")
        # draft_i_start = time.time()
        # Pass the draft KVCache list to the model during drafting
        with model.self_draft(dynamic_skip_mask=current_draft_skip_mask):
            draft_outputs = model(  # Call the main model forward
                input_ids=next_draft_input_ids,
                past_key_values=draft_kv_cache_list,  # Use draft cache list
                # past_key_values=past_key_values,
                output_hidden_states=False,
            )
        # draft_i_end_time = time.time()
        # logging.info(
        #     f"Drafting foward step {draft_step_idx + 1} time: {draft_i_end_time - draft_i_start:.4f}s"
        # )
        draft_logits = draft_outputs.logits[:, -1, :]  # Logits for next token
        # draft_kv_cache_list is automatically updated by the forward pass
        # (DET check and sampling logic unchanged)
        if logits_processor:
            draft_logits_processed = logits_processor(
                torch.tensor(input_ids_list, device=device), draft_logits
            )  # Pass full history?
        else:
            draft_logits_processed = draft_logits
        draft_probs = torch.softmax(draft_logits_processed, dim=-1)
        top1_prob = torch.max(draft_probs, dim=-1).values.item()
        if logits_processor:
            next_token = torch.multinomial(draft_probs, num_samples=1)
        else:
            next_token = torch.argmax(draft_logits_processed, dim=-1, keepdim=True)
        draft_tokens.append(next_token.item())
        next_draft_input_ids = next_token
        if top1_prob < DET and len(draft_tokens) > 0:
            break

        if next_token.item() == eos_token_id:
            continue_hc = False
            break

        if len(draft_tokens) + new_token_num >= max_new_tokens - 2:
            continue_hc = False
            break
    # PLD HC
    if HC and continue_hc:
        draft_ids = torch.tensor(
            [input_ids_list[0] + draft_tokens], device=device
        )
        max_new_draft = max_new_tokens - len(draft_tokens) - new_token_num -2
        pld_candidate_tokens = find_candidate_pred_tokens(
            draft_ids,
            max_ngram_size=5,
            num_pred_tokens=min(10, max_new_draft),
        )
        draft_tokens += pld_candidate_tokens.tolist()
        

    # # Clone KV cache specifically for this drafting phase if needed, or manage carefully.
    # draft_past_key_values = clone_past_key_values(
    #     model, past_key_values_data, current_length_data
    # )
    # draft_tokens, draft_probs, draft_full_probs, top1_probs_list = [], [], [], []
    # current_input_ids = input_ids
    # for step_draft in range(max_step_draft):
    #     # Apply the DYNAMIC skip mask for this draft using a context manager
    #     # This requires modifying the model's LlamaDecoderLayer or the context manager
    #     # to accept and use 'draft_skip_mask' temporarily.
    #     with model.self_draft(dynamic_skip_mask=draft_skip_mask):
    #         draft_outputs = model.model(
    #             input_ids=current_input_ids,
    #             past_key_values=draft_past_key_values,
    #             position_ids=position_ids,
    #             output_hidden_states=False,  # Don't need hidden states during draft
    #             use_cache=True,  # Essential for autoregressive drafting
    #         )
    #     # draft_outputs includes logits and updated past_key_values for the draft model
    #     current_draft_logits = model.lm_head(draft_outputs[0])
    #     draft_past_key_values = (
    #         draft_outputs.past_key_values
    #     )  # Use the updated KV cache
    #     # --- Sampling (same as swift_draft) ---
    #     if logits_processor is not None:
    #         topk_index, topk_prob, full_prob = sample(
    #             current_draft_logits[:, -1, :], logits_processor, k=TOPK
    #         )
    #         next_token = topk_index[:, 0].unsqueeze(-1)  # Shape (batch=1, 1)
    #     else:
    #         top = torch.topk(current_draft_logits[:, -1, :], TOPK, dim=-1)
    #         topk_index, topk_prob = top.indices, top.values
    #         next_token = topk_index[:, 0].unsqueeze(-1)  # Shape (batch=1, 1)
    #         full_prob = None  # No full prob distribution in greedy
    #     draft_tokens.append(topk_index)
    #     draft_probs.append(topk_prob)
    #     draft_full_probs.append(full_prob)
    #     # --- Draft Exiting Threshold (DET) Check ---
    #     origin_draft_probs = torch.softmax(current_draft_logits[:, -1, :], dim=-1)
    #     argmax_prob = torch.gather(origin_draft_probs, -1, next_token).squeeze(-1)
    #     current_threshold = argmax_prob.item()
    #     top1_probs_list.append(current_threshold)
    #     # Update input for next draft step
    #     current_input_ids = next_token
    #     if position_ids is not None:
    #         position_ids = position_ids[..., -1:] + 1  # Increment position id
    #     if (
    #         current_threshold < stop_threshold
    #         or new_token_num + step_draft + 2 >= max_new_tokens
    #     ):
    #         break
    # # Ensure outputs are tensors even if loop finishes early or runs 0 times
    # if not draft_tokens:
    #     # Return empty tensors of appropriate type/device if no tokens were drafted
    #     empty_long = torch.tensor([], dtype=torch.long, device=input_ids.device)
    #     empty_float = torch.tensor([], dtype=model.dtype, device=input_ids.device)
    #     return (empty_long, empty_float, []), []
    # return (
    #     torch.cat(draft_tokens, dim=0),
    #     torch.cat(draft_probs, dim=0),
    #     draft_full_probs,
    # ), top1_probs_list


def reset_past_key_values(past_key_values):
    """
    Resets the current lengths in the past key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - past_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(past_key_values)):
        for j in range(2):
            past_key_values[i][j].current_length.fill_(0)
    return past_key_values


# --- Core CLaSp DP Algorithm ---
def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    # Ensure vectors are normalized for efficiency if done frequently
    vec1_norm = vec1 / torch.linalg.norm(vec1)
    vec2_norm = vec2 / torch.linalg.norm(vec2)
    return torch.dot(vec1_norm, vec2_norm)


def normalize_tensor(tensor):
    """Normalizes a tensor (vector or rows of a matrix)."""
    if tensor.ndim == 1:
        norm = torch.linalg.norm(tensor)
        return tensor / norm if norm > 0 else tensor
    elif tensor.ndim == 2:
        norm = torch.linalg.norm(tensor, dim=1, keepdim=True)
        return tensor / torch.where(
            norm > 0, norm, torch.tensor(1.0, device=tensor.device)
        )
    else:
        # Handle batch dim if needed, assuming (batch, seq, hidden) -> (batch*seq, hidden)
        original_shape = tensor.shape
        tensor_2d = tensor.view(-1, original_shape[-1])
        norm = torch.linalg.norm(tensor_2d, dim=1, keepdim=True)
        normalized_2d = tensor_2d / torch.where(
            norm > 0, norm, torch.tensor(1.0, device=tensor.device)
        )
        return normalized_2d.view(original_shape)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask_id(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Create a causal mask for bi-directional self-attention.

    Args:
        input_ids_shape (torch.Size): The shape of input_ids tensor, typically (batch_size, tgt_len).
        dtype (torch.dtype): The data type of the mask.
        device (torch.device): The device on which the mask will be placed.
        past_key_values_length (int, optional): The length of past key values. Default is 0.

    Returns:
        torch.Tensor: The causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # Set the diagonal to 0, allowing each position to attend to itself
    mask.fill_diagonal_(0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


@torch.no_grad()
def CLaSp_Skip_Layer_Strategy_SeqParallel(
    L, M, hidden_states_H, model_layers, past_key_values, device
):
    """
    Implements the CLaSp DP algorithm with Sequence Parallel optimization.

    Args:
        L: Total number of decoder layers.
        M: Target number of layers to skip.
        hidden_states_H: List/Tuple of hidden states {h_0, ..., h_L} from verify pass.
        model_layers: The list/nn.ModuleList of the verify model's layers.
        past_key_values: The past key values for the model.
        device: The torch device.

    Returns:
        S: A boolean torch tensor of size L indicating skipped layers.
    """
    model_dtype = next(
        model_layers[0].parameters()
    ).dtype  # Get model dtype dynamically
    # d = hidden_states_H[0].shape[-1]
    past_key_values_length = 0
    for past_key_value in past_key_values:
        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            break

    g = {}
    parent = {}
    g[(0, 0)] = hidden_states_H[0].to(device)

    # --- Dynamic Programming with Sequence Parallel ---
    for i in range(1, L + 1):
        g[(i, 0)] = hidden_states_H[i].to(device)
        parent[(i, 0)] = (i - 1, 0, False)  # No skip for j=0
        layer_func = model_layers[i - 1]
        max_prev_skips = min(i - 1, M)

        # --- Parallel Layer Computation ---
        # 1. Identify states needing computation by layer i-1
        states_to_process_indices_j = []
        states_to_process_tensors = []
        for j_prev in range(1, max_prev_skips + 1):
            if (i - 1, j_prev) in g:
                states_to_process_indices_j.append(j_prev)
                states_to_process_tensors.append(g[(i - 1, j_prev)])

        computed_states_map = {}  # Map j -> computed state f_{i-1}(g(i-1, j))

        if states_to_process_tensors:
            # 2. Prepare input sequence and attention mask
            num_parallel_j = len(states_to_process_indices_j)
            # Stack along a new dimension, then unsqueeze for batch dim = 1
            # Input shape expected by layer: (batch_size, seq_len, hidden_dim)
            input_sequence = torch.stack(states_to_process_tensors, dim=0).unsqueeze(
                0
            )  # Shape: (1, num_parallel_j, d)

            # 3. Create the "special mask matrix"
            # Mask needs shape (batch_size, num_heads, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
            # Using additive mask (0 for attend, -inf for mask)
            attn_mask = _make_causal_mask_id(
                input_sequence.shape[:2],
                dtype=model_dtype,
                device=device,
                past_key_values_length=past_key_values_length,
            )

            # 4. Call the layer ONCE with the prepared sequence and mask
            # reuse KV cache for all parallel computations
            past_key_value = (
                past_key_values[i - 1] if past_key_values is not None else None
            )
            # position ids all set to past_key_values_length
            position_ids = torch.full(
                (1, num_parallel_j), past_key_values_length, device=device
            ).long()
            layer_outputs = layer_func(
                input_sequence,
                attention_mask=attn_mask,
                past_key_value=past_key_value,
                position_ids=position_ids,
                output_attentions=False,
            )
            # Output hidden state shape: (1, num_parallel_j, d)
            output_states_sequence = layer_outputs[0].squeeze(
                0
            )  # Shape: (num_parallel_j, d)

            # 5. Store the computed states mapped back to their original 'j' index
            for k, j_orig in enumerate(states_to_process_indices_j):
                computed_states_map[j_orig] = output_states_sequence[k]
        # --- End Parallel Layer Computation ---

        # max_j = min(i - 1, M)
        # --- Update DP table using pre-computed states ---
        for j in range(1, max_prev_skips + 1):
            norm_hi = None
            # Option 1: Arrive via NOT skipping layer i-1 (from state (i-1, j))
            state_if_not_skipped = None
            sim_if_not_skipped = -float("inf")
            # Check if the required state was computed in the parallel step
            if j in computed_states_map:
                norm_hi = normalize_tensor(g[(i, 0)])
                state_if_not_skipped = computed_states_map[j]
                sim_if_not_skipped = cosine_similarity(
                    normalize_tensor(state_if_not_skipped), norm_hi
                )

            # Option 2: Arrive via SKIPPING layer i-1 (from state (i-1, j-1))
            state_if_skipped = None
            sim_if_skipped = -float("inf")
            if (i - 1, j - 1) in g:
                norm_hi = normalize_tensor(g[(i, 0)]) if norm_hi is None else norm_hi
                state_if_skipped = g[(i - 1, j - 1)]
                sim_if_skipped = cosine_similarity(
                    normalize_tensor(state_if_skipped), norm_hi
                )

            # Decide based on cosine similarity
            # Important: Handle cases where one path is impossible (e.g., j=0 can't come from j-1)
            if state_if_not_skipped is None and state_if_skipped is None:
                continue  # This state (i, j) is unreachable

            if state_if_not_skipped is not None and (
                state_if_skipped is None or sim_if_not_skipped >= sim_if_skipped
            ):
                # Choose the "not skipped" path if it's possible and better/equal, OR if "skipped" path is impossible
                g[(i, j)] = state_if_not_skipped
                parent[(i, j)] = (i - 1, j, False)
            elif state_if_skipped is not None:
                # Choose the "skipped" path if it's possible and strictly better, OR if "not skipped" path was impossible
                g[(i, j)] = state_if_skipped
                parent[(i, j)] = (i - 1, j - 1, True)
        # When skipping all:
        if i <= M:
            # If we can skip all layers, we can just take the last layer's output
            g[(i, i)] = g[(i - 1, i - 1)]
            parent[(i, i)] = (i - 1, i - 1, True)

    # --- Backtracking ---
    S = torch.zeros(L, dtype=torch.bool, device=device)
    current_i, current_j = L, M

    if (L, M) not in g:
        # Fallback logic to find best available j_final
        best_final_j = -1
        max_final_sim = -float("inf")
        h_L = hidden_states_H[L].to(device)
        for j_final in range(M + 1):
            if (L, j_final) in g:
                sim = cosine_similarity(
                    normalize_tensor(g[(L, j_final)]), normalize_tensor(h_L)
                )
                if sim > max_final_sim:
                    max_final_sim = sim
                    best_final_j = j_final
        if best_final_j != -1:
            current_j = best_final_j
            logging.infor(
                f"CLaSp DP: Target skips M={M} not reached. Using best j={current_j} at final layer."
            )
        else:
            logging.infor(
                "CLaSp DP: No valid path found in DP table during backtracking."
            )
            return S

    while current_i > 0:
        if (current_i, current_j) not in parent:
            logging.info(
                f"CLaSp DP: Backtracking error - state ({current_i}, {current_j}) has no parent."
            )
            break
        prev_i, prev_j, was_skipped = parent[(current_i, current_j)]
        if was_skipped:
            S[current_i - 1] = True
        current_i, current_j = prev_i, prev_j

    # Sanity check for number of skips
    actual_skips = torch.sum(S).item()
    if actual_skips > M:  # Check if backtracking selected more skips than allowed
        logging.info(
            f"CLaSp DP Backtracking resulted in {actual_skips} skips, but target was {M}. Check logic."
        )
        # Optional: Implement a correction mechanism if this happens, e.g., un-skip layers
        # with lowest impact based on some heuristic, though this indicates a DP logic issue.

    return S
