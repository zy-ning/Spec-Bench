# import copy
import logging

# import numpy as np
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .kv_cache import clone_past_key_values

TOPK = 10  # topk


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


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def clasp_verify(
    model,
    input_ids=None,
    past_key_values=None,
    position_ids=None,
):
    """
    Verify the clasp structure using the provided model and input.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (LLM): The model containing the full LLM model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.
    - position_ids (torch.Tensor): Positional IDs associated with the clasp structure.

    Returns:
    - outputs (tuple): Contains the outputs from the model.
    - orig (torch.Tensor): Original logits from the full model.
    """
    with torch.inference_mode():
        # Pass input through the base model
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        orig = model.lm_head(outputs[0])

    return outputs, orig


def sample(logits, logits_processor, k=1):
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
    input_ids=None,
    draft_skip_mask=None,  # The dynamically calculated skip mask S
    new_token_num=0,
    past_key_values_data=None,  # Use CLONED KV cache for drafting
    current_length_data=None,
    max_new_tokens=1024,
    position_ids=None,
    max_step_draft=25,
    logits_processor=None,
    stop_threshold=0.8,  # Corresponds to Draft-Exiting Threshold (DET)
):
    """
    Draft new tokens usißng the CLaSp dynamically determined skip mask.
    Args:
        model: The LLM model instance.
        input_ids: Starting token(s) for drafting.
        draft_skip_mask: Boolean tensor (size L) indicating which layers to skip for THIS draft.
    Returns:
        Tuple: (draft_tokens, draft_probs, draft_full_probs), top1_probs_list
    """
    # Clone KV cache specifically for this drafting phase if needed, or manage carefully.
    draft_past_key_values = clone_past_key_values(
        model, past_key_values_data, current_length_data
    )
    draft_tokens, draft_probs, draft_full_probs, top1_probs_list = [], [], [], []
    current_input_ids = input_ids
    for step_draft in range(max_step_draft):
        # Apply the DYNAMIC skip mask for this draft using a context manager
        # This requires modifying the model's LlamaDecoderLayer or the context manager
        # to accept and use 'draft_skip_mask' temporarily.
        with model.self_draft(dynamic_skip_mask=draft_skip_mask):
            draft_outputs = model.model(
                input_ids=current_input_ids,
                past_key_values=draft_past_key_values,
                position_ids=position_ids,
                output_hidden_states=False,  # Don't need hidden states during draft
                use_cache=True,  # Essential for autoregressive drafting
            )
        # draft_outputs includes logits and updated past_key_values for the draft model
        current_draft_logits = model.lm_head(draft_outputs[0])
        draft_past_key_values = (
            draft_outputs.past_key_values
        )  # Use the updated KV cache
        # --- Sampling (same as swift_draft) ---
        if logits_processor is not None:
            topk_index, topk_prob, full_prob = sample(
                current_draft_logits[:, -1, :], logits_processor, k=TOPK
            )
            next_token = topk_index[:, 0].unsqueeze(-1)  # Shape (batch=1, 1)
        else:
            top = torch.topk(current_draft_logits[:, -1, :], TOPK, dim=-1)
            topk_index, topk_prob = top.indices, top.values
            next_token = topk_index[:, 0].unsqueeze(-1)  # Shape (batch=1, 1)
            full_prob = None  # No full prob distribution in greedy
        draft_tokens.append(topk_index)
        draft_probs.append(topk_prob)
        draft_full_probs.append(full_prob)
        # --- Draft Exiting Threshold (DET) Check ---
        origin_draft_probs = torch.softmax(current_draft_logits[:, -1, :], dim=-1)
        argmax_prob = torch.gather(origin_draft_probs, -1, next_token).squeeze(-1)
        current_threshold = argmax_prob.item()
        top1_probs_list.append(current_threshold)
        # Update input for next draft step
        current_input_ids = next_token
        if position_ids is not None:
            position_ids = position_ids[..., -1:] + 1  # Increment position id
        if (
            current_threshold < stop_threshold
            or new_token_num + step_draft + 2 >= max_new_tokens
        ):
            break
    # Ensure outputs are tensors even if loop finishes early or runs 0 times
    if not draft_tokens:
        # Return empty tensors of appropriate type/device if no tokens were drafted
        empty_long = torch.tensor([], dtype=torch.long, device=input_ids.device)
        empty_float = torch.tensor([], dtype=model.dtype, device=input_ids.device)
        return (empty_long, empty_float, []), []
    return (
        torch.cat(draft_tokens, dim=0),
        torch.cat(draft_probs, dim=0),
        draft_full_probs,
    ), top1_probs_list

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


@torch.no_grad()
def CLaSp_Skip_Layer_Strategy_SeqParallel(L, M, hidden_states_H, model_layers, device):
    """
    Implements the CLaSp DP algorithm with Sequence Parallel optimization.

    Args:
        L: Total number of decoder layers.
        M: Target number of layers to skip.
        hidden_states_H: List/Tuple of hidden states {h_0, ..., h_L} from verify pass.
        model_layers: The list/nn.ModuleList of the verify model's layers.
        device: The torch device.

    Returns:
        S: A boolean torch tensor of size L indicating skipped layers.
    """
    model_dtype = next(
        model_layers[0].parameters()
    ).dtype  # Get model dtype dynamically
    # d = hidden_states_H[0].shape[-1]
    g = {}
    parent = {}
    g[(0, 0)] = hidden_states_H[0].to(device)

    # --- Dynamic Programming with Sequence Parallel ---
    for i in range(1, L + 1):
        h_i = hidden_states_H[i].to(device)
        layer_func = model_layers[i - 1]
        max_prev_skips = min(i - 1, M)

        # --- Parallel Layer Computation ---
        # 1. Identify states needing computation by layer i-1
        states_to_process_indices_j = []
        states_to_process_tensors = []
        for j_prev in range(max_prev_skips + 1):
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

            # 3. Create the "special mask matrix" (identity attention)
            # Mask needs shape (batch_size, num_heads, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
            # Using additive mask (0 for attend, -inf for mask)
            attn_mask = torch.full(
                (1, 1, num_parallel_j, num_parallel_j),
                dtype=model_dtype,  # Ensure mask has same dtype as model
                device=device,
                fill_value=-torch.inf,  # Mask everything initially
                # Use -torch.inf or large negative number like -1e9 depending on implementation
            )
            # Allow self-attention only by setting diagonal to 0
            attn_mask[:, :, range(num_parallel_j), range(num_parallel_j)] = 0

            # 4. Call the layer ONCE with the prepared sequence and mask
            # We don't need KV cache or attention outputs for DP state calculation
            layer_outputs = layer_func(
                input_sequence,
                attention_mask=attn_mask,
                use_cache=False,
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

        # --- Update DP table using pre-computed states ---
        for j in range(min(i, M) + 1):
            # Option 1: Arrive via NOT skipping layer i-1 (from state (i-1, j))
            state_if_not_skipped = None
            sim_if_not_skipped = -float("inf")
            # Check if the required state was computed in the parallel step
            if j in computed_states_map:
                state_if_not_skipped = computed_states_map[j]
                sim_if_not_skipped = cosine_similarity(
                    normalize_tensor(state_if_not_skipped), normalize_tensor(h_i)
                )

            # Option 2: Arrive via SKIPPING layer i-1 (from state (i-1, j-1))
            state_if_skipped = None
            sim_if_skipped = -float("inf")
            if j > 0 and (i - 1, j - 1) in g:
                state_if_skipped = g[(i - 1, j - 1)]
                sim_if_skipped = cosine_similarity(
                    normalize_tensor(state_if_skipped), normalize_tensor(h_i)
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
            logging.warning(
                f"CLaSp DP: Target skips M={M} not reached. Using best j={current_j} at final layer."
            )
        else:
            logging.error(
                "CLaSp DP: No valid path found in DP table during backtracking."
            )
            return S

    while current_i > 0:
        if (current_i, current_j) not in parent:
            logging.error(
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
        logging.warning(
            f"CLaSp DP Backtracking resulted in {actual_skips} skips, but target was {M}. Check logic."
        )
        # Optional: Implement a correction mechanism if this happens, e.g., un-skip layers
        # with lowest impact based on some heuristic, though this indicates a DP logic issue.

    return S
