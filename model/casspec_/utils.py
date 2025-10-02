# import copy
import copy
import logging
import random

import numpy as np

# import numpy as np
import torch

# import torch.nn as nn
# import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# from transformers.generation.utils import _crop_past_key_values
from model.pld.pld import find_candidate_pred_tokens

from .kv_cache import clone_past_key_values

TOPK = 10


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


def sample(
    logits: torch.Tensor,
    logits_processor: LogitsProcessorList,
    all_logits: bool = False,
    k: int = 1,
):
    """
    Sample from the provided logits using the specified processor.

    Args:
    - logits (torch.Tensor): Logits to sample from. Expected shapes:
        - If all_logits=False: (batch_size, vocab_size) or (batch_size, seq_len, vocab_size)
                                (if 3D, only logits for the last sequence position are used).
                                Can also be (vocab_size) for a single unbatched distribution.
        - If all_logits=True: (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
                                (if 2D, batch_size is assumed to be 1).
    - logits_processor (LogitsProcessorList): Processor to use for sampling.
    - all_logits (bool): If True, sample for all sequence positions.
    - k (int): Number of samples to generate per distribution. Must be >= 1.

    Returns:
    - sampled_indices (torch.Tensor): Indices of the sampled tokens.
    - sampled_probs (torch.Tensor): Conditional probabilities of the sampled tokens.
                                     P(sample_m | sample_1, ..., sample_{m-1}).
    - probabilities (torch.Tensor): Probabilities of all tokens (after processing by logits_processor,
                                     before multinomial sampling).
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    initial_ndim = logits.ndim
    device = logits.device

    # 1. Prepare logits_to_process based on all_logits flag and input shape
    #    Also determine the prefix shape for the output tensors.
    if not all_logits:
        if initial_ndim == 3:  # (batch_size, seq_len, vocab_size)
            batch_size, _, vocab_size = logits.shape
            logits_to_process = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            output_prefix_shape = (batch_size,)
        elif initial_ndim == 2:  # (batch_size, vocab_size)
            batch_size, vocab_size = logits.shape
            logits_to_process = logits
            output_prefix_shape = (batch_size,)
        elif initial_ndim == 1:  # (vocab_size) - unbatched
            vocab_size = logits.shape[0]
            logits_to_process = logits.unsqueeze(0)  # Shape: (1, vocab_size)
            output_prefix_shape = (1,)  # Will be squeezed later if initial_ndim == 1
        else:
            raise ValueError(
                f"Unsupported logits shape for all_logits=False: {logits.shape}"
            )
    else:  # all_logits is True
        if initial_ndim == 3:  # (batch_size, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = logits.shape
            logits_to_process = logits.reshape(
                -1, vocab_size
            )  # Shape: (batch_size * seq_len, vocab_size)
            output_prefix_shape = (batch_size, seq_len)
        elif (
            initial_ndim == 2
        ):  # (seq_len, vocab_size) - assumed batch_size=1 for output shaping
            seq_len, vocab_size = logits.shape
            logits_to_process = logits.reshape(
                -1, vocab_size
            )  # Shape: (seq_len, vocab_size)
            output_prefix_shape = (
                seq_len,
            )  # Output will be (seq_len, k), (seq_len, vocab_size)
            # This implies batch_size=1 was "added and removed"
        else:
            raise ValueError(
                f"Unsupported logits shape for all_logits=True: {logits.shape}. Expected 2D or 3D."
            )

    # 2. Process logits using the provided logits_processor
    processed_logits = logits_processor(None, logits_to_process)

    # 3. Calculate probabilities for all tokens (after processing)
    full_probabilities_flat = torch.nn.functional.softmax(processed_logits, dim=-1)

    # 4. Sample k indices using multinomial sampling without replacement
    sampled_indices_flat = torch.multinomial(
        full_probabilities_flat, k, replacement=False
    )

    # 5. Gather the probabilities of the sampled tokens from the full_probabilities distribution
    sampled_probs_raw_flat = torch.gather(
        full_probabilities_flat, -1, sampled_indices_flat
    )

    # 6. Adjust sampled_probs for sequential sampling without replacement (conditional probabilities)
    # This calculates P(sample_m | sample_1, ..., sample_{m-1})
    cumulative_sum = torch.cumsum(sampled_probs_raw_flat, dim=-1)
    cumulative_sum_shifted = torch.cat(
        (
            torch.zeros(cumulative_sum.shape[0], 1, device=device),
            cumulative_sum[:, :-1],
        ),
        dim=-1,
    )

    conditional_sampled_probs_flat = sampled_probs_raw_flat / (
        1 - cumulative_sum_shifted
    )

    # Handle numerical issues as in the original snippet
    conditional_sampled_probs_flat[torch.isinf(conditional_sampled_probs_flat)] = -1.0
    conditional_sampled_probs_flat[torch.isnan(conditional_sampled_probs_flat)] = -1.0

    # Clamp to [0,1] range. Values set to -1 become 0.
    final_sampled_probs_flat = torch.clamp(
        conditional_sampled_probs_flat, min=0.0, max=1.0
    )

    # 7. Reshape outputs to their final forms
    if not all_logits:
        # If the original input was 1D (vocab_size), squeeze the batch dimension we added.
        if initial_ndim == 1:
            final_sampled_indices = sampled_indices_flat.squeeze(0)
            final_sampled_probs = final_sampled_probs_flat.squeeze(0)
            final_full_probabilities = full_probabilities_flat.squeeze(0)
        else:
            # Shape: (batch_size, k) or (batch_size, vocab_size)
            final_sampled_indices = sampled_indices_flat
            final_sampled_probs = final_sampled_probs_flat
            final_full_probabilities = full_probabilities_flat
    else:  # all_logits was True
        # Reshape flat outputs to include batch_size and seq_len dimensions (or just seq_len if input was 2D).
        # output_prefix_shape is (batch_size, seq_len) or (seq_len,)
        final_sampled_indices = sampled_indices_flat.view(*output_prefix_shape, k)
        final_sampled_probs = final_sampled_probs_flat.view(*output_prefix_shape, k)
        final_full_probabilities = full_probabilities_flat.view(
            *output_prefix_shape, vocab_size
        )

    return final_sampled_indices, final_sampled_probs, final_full_probabilities


# @torch.no_grad()
def cassepc_draft(
    model,
    device,
    input_ids=None,
    new_token_num=0,
    past_key_values_data=None,
    current_length_data=None,
    logits_processor=None,
    max_new_tokens=1024,
    eos_token_id=None,
    current_draft_skip_mask=None,
    K=25,
    DET=0.7,
):
    # draft_tokens = []

    # first_draft_input_ids = torch.tensor(
    #     [input_ids_list[0][-1]], device=device
    # ).unsqueeze(0)
    first_draft_input_ids = input_ids[:, -1].unsqueeze(0)
    is_eos = False

    if current_draft_skip_mask is not None:
        draft_tree_logits, top1_prob, is_eos = clasp_draft(
            model,
            first_draft_input_ids,
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            logits_processor=logits_processor,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            current_draft_skip_mask=current_draft_skip_mask,
            K=K,
            DET=DET,
            # VC=True,
            # all_ids=input_ids,
        )
        # if draft_tree_logits is None:
        #     return [], [], is_eos
        # draft_tokens_t = get_vanilla_tokens(draft_tree_logits)
        # draft_tokens.extend(draft_tokens_t.tolist())

    # if not is_eos:
    #     all_ids = torch.cat((all_ids, draft_tokens_t.unsqueeze(0)), dim=-1)
    #     max_new_draft = max_new_tokens - len(draft_tokens) - new_token_num -2
    #     if max_new_draft > 1:
    #         pld_candidate_tokens = find_candidate_pred_tokens(
    #             all_ids,
    #             max_ngram_size=3,
    #             num_pred_tokens=min(5, max_new_draft),
    #         )
    #         if len(pld_candidate_tokens) > 0:
    #             draft_tokens += pld_candidate_tokens.tolist()

    return draft_tree_logits, top1_prob, is_eos


@torch.no_grad()
def clasp_draft(
    model,
    input_ids,
    past_key_values_data=None,
    current_length_data=None,
    new_token_num=0,
    logits_processor=None,
    eos_token_id=None,
    max_new_tokens=1024,
    current_draft_skip_mask=None,
    K=2,
    DET=0.7,
    VC=False,
    all_ids=None,
):
    ss_token, ss_prob, ss_op, top1_prob = [], [], [], []
    next_draft_input_ids = input_ids
    is_eos = False
    max_new_draft = max_new_tokens - new_token_num
    K = min(K, max_new_draft)
    if K < 1:
        raise ValueError("K must be at least 1.")

    # Use a *cloned* KV cache structure for drafting
    draft_past_key_values_data_list = [d.clone() for d in past_key_values_data]
    draft_current_length_data = current_length_data.clone()
    draft_kv_cache_list = clone_past_key_values(
        model, draft_past_key_values_data_list, draft_current_length_data
    )

    if VC and all_ids is not None and K > 1:
        turns = int(np.ceil(float(K) / 1.5))
        kv_cur_len = draft_current_length_data[0].item()
        for i in range(turns):
            pld_candidate_tokens = find_candidate_pred_tokens(
                all_ids,
                max_ngram_size=5,
                num_pred_tokens=min(5, max_new_draft),
            )
            cand_len = pld_candidate_tokens.size(-1)
            next_draft_input_ids = all_ids[:, -1].unsqueeze(0)
            if cand_len > 0:
                # cat next_draft_input_ids with pld_candidate_tokens
                next_draft_input_ids = torch.cat(
                    (next_draft_input_ids, pld_candidate_tokens.unsqueeze(0)), dim=-1
                )
            # --verify as drafting--
            with model.self_draft(dynamic_skip_mask=current_draft_skip_mask):
                draft_outputs = model(
                    input_ids=next_draft_input_ids,
                    past_key_values=draft_kv_cache_list,
                    output_hidden_states=False,
                )
            new_logits = draft_outputs.logits[:, -cand_len - 1 :]
            if logits_processor is not None:
                topk_index, topk_prob, op = sample(
                    new_logits, logits_processor, all_logits=True, k=TOPK
                )
            else:
                top = torch.topk(new_logits, TOPK, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                op = None
            selected_tokens = topk_index[:, :, 0]
            candidate_new_tokens = next_draft_input_ids[:, -cand_len:]
            if cand_len > 0:
                vc_accepted_len = (
                    (~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1)
                    < 1
                ).sum()
            else:
                vc_accepted_len = 0
            # accept_rate = vc_accepted_len / cand_len if cand_len > 0 else 0
            overflow_len = vc_accepted_len - max_new_draft
            if overflow_len > 0:
                vc_accepted_len = vc_accepted_len - overflow_len
            max_new_draft -= vc_accepted_len + 1
            valid_tokens = selected_tokens[:, : vc_accepted_len + 1]
            all_ids = torch.cat((all_ids, valid_tokens[:, 1:]), dim=-1)
            # after unsqueeze, the shape is (1, batch_size, seq_len, x), so unbind(2)
            ss_token.extend(
                list(topk_index[:, : vc_accepted_len + 1].unsqueeze(0).unbind(2))
            )
            ss_prob.extend(
                list(topk_prob[:, : vc_accepted_len + 1].unsqueeze(0).unbind(2))
            )
            ss_op.extend(
                list(topk_prob[:, : vc_accepted_len + 1].unsqueeze(0).unbind(2))
            )
            origin_draft_probs = new_logits.softmax(-1)[:, : vc_accepted_len + 1]
            argmax_prob = torch.gather(
                origin_draft_probs, -1, input_ids.unsqueeze(-1)
            ).squeeze(-1)
            top1_prob.extend(argmax_prob.squeeze(0).tolist())
            min_tpo1_prob = torch.min(argmax_prob).item()
            # --update past_key_values--
            new_cache_size = kv_cur_len + vc_accepted_len + 1
            draft_current_length_data.fill_(new_cache_size)

            if eos_token_id in valid_tokens.squeeze(0).tolist():
                is_eos = True
                break
            if min_tpo1_prob < DET:
                break
            if overflow_len > 0:
                break
    else:
        for i in range(K):
            with model.self_draft(dynamic_skip_mask=current_draft_skip_mask):
                draft_outputs = model(
                    input_ids=next_draft_input_ids,
                    past_key_values=draft_kv_cache_list,
                    output_hidden_states=False,
                )
            draft_logits = draft_outputs.logits
            if logits_processor is not None:
                topk_index, topk_prob, op = sample(
                    draft_logits, logits_processor, k=TOPK
                )
                next_draft_input_ids = topk_index[:, 0].unsqueeze(0)
            else:
                top = torch.topk(draft_logits, TOPK, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                next_draft_input_ids = topk_index[:, :, 0]
                op = None
            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)
            origin_draft_probs = draft_logits.softmax(-1)
            argmax_prob = torch.gather(
                origin_draft_probs, -1, input_ids.unsqueeze(-1)
            ).squeeze(-1)
            current_threshold = argmax_prob.item()
            top1_prob.append(current_threshold)
            if next_draft_input_ids.item() == eos_token_id:
                is_eos = True
                break
            if current_threshold < DET and i > 0:
                break
    return (torch.cat(ss_token), torch.cat(ss_prob), ss_op), top1_prob, is_eos


def get_vanilla_tokens(
    draft_tree_logits,
):
    return draft_tree_logits[0][:, -1, 0]


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


# --- Tree (SWIFT) ---
def get_choices_list(prob_list, logits_processor=None):
    """
    Generate tree choices list based on the provided confidence.

    Parameters:
    - prob_list (list): List of probabilities.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.

    Returns:
    - list: A nested list containing choices based on the probabilities.
    """
    choices_list = []
    if logits_processor is not None:
        candidate_set = [1, 3, 5, 10]
    else:
        candidate_set = [3, 5, 8, 10]
    for idx, item in enumerate(prob_list):
        if item is None:
            candidate_num = 1
        elif item > 0.95:
            candidate_num = candidate_set[0]
        elif item > 0.8:
            candidate_num = candidate_set[1]
        elif item > 0.5:
            candidate_num = candidate_set[2]
        else:
            candidate_num = candidate_set[3]
        choices_list.extend([[0] * idx + [i] for i in range(candidate_num)])
    return choices_list


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


def generate_swift_buffers(swift_choices, device="cuda"):
    """
    Generate buffers for the swift structure based on the provided choices.

    Parameters:
    - swift_choices (list): A nested list representing tree in the swift structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".

    Returns:
    - dict: A dictionary containing buffers related to the swift structure.
    """
    # Sort the swift_choices based on their lengths and then their values
    sorted_swift_choices = sorted(swift_choices, key=lambda x: (len(x), x))
    swift_len = len(sorted_swift_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_swift_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Create the attention mask for swift
    swift_attn_mask = torch.eye(swift_len, swift_len)
    swift_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_swift_choice = sorted_swift_choices[start + j]
            # retrieve ancestor position
            if len(cur_swift_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_swift_choice) - 1):
                ancestor_idx.append(
                    sorted_swift_choices.index(cur_swift_choice[: c + 1]) + 1
                )
            swift_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the swift structure
    swift_tree_indices = torch.zeros(swift_len, dtype=torch.long)
    swift_p_indices = [0 for _ in range(swift_len - 1)]
    swift_b_indices = [[] for _ in range(swift_len - 1)]
    swift_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        b = []
        for j in range(depth_counts[i]):
            cur_swift_choice = sorted_swift_choices[start + j]
            swift_tree_indices[start + j + 1] = cur_swift_choice[-1] + TOPK * i + 1
            swift_p_indices[start + j] = 0
            if len(b) > 0:
                swift_b_indices[start + j] = copy.deepcopy(b)
            else:
                swift_b_indices[start + j] = []
            b.append(cur_swift_choice[-1] + TOPK * i + 1)
        start += depth_counts[i]

    # Generate position IDs for the swift structure
    swift_p_indices = [-1] + swift_p_indices
    swift_position_ids = torch.zeros(swift_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        swift_position_ids[start + 1 : start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for swift structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_swift_choices)):
        cur_swift_choice = sorted_swift_choices[-i - 1]
        retrieve_indice = []
        if cur_swift_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_swift_choice)):
                retrieve_indice.append(
                    sorted_swift_choices.index(cur_swift_choice[: c + 1])
                )
                retrieve_paths.append(cur_swift_choice[: c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat(
        [
            torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long),
            retrieve_indices,
        ],
        dim=1,
    )

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    swift_p_indices = torch.tensor(swift_p_indices)
    swift_p_indices_new = swift_p_indices[retrieve_indices]
    swift_p_indices_new = swift_p_indices_new.tolist()

    swift_b_indices = [[]] + swift_b_indices
    swift_b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = swift_b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(swift_tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        swift_b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    swift_buffers = {
        "swift_attn_mask": swift_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": swift_tree_indices,
        "swift_position_ids": swift_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    swift_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in swift_buffers.items()
    }
    swift_buffers["p_indices"] = swift_p_indices_new
    swift_buffers["b_indices"] = swift_b_indices_new
    return swift_buffers


def generate_candidates(
    swift_logits, tree_indices, retrieve_indices, sample_token, logits_processor
):
    """
    Generate candidates based on provided logits and indices.

    Parameters:
    - swift_logits (torch.Tensor): Logits associated with the swift structure.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    - sample_token (torch.Tensor): Token sampled from probability distribution.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.

    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
    sample_token = sample_token.to(tree_indices.device)

    # Greedy decoding: Select the most probable candidate from the original logits.
    candidates_logit = sample_token[0]

    # Extract the TOPK candidates from the swift logits.
    candidates_swift_logits = swift_logits[0]

    # Combine the selected candidate from the original logits with the topk swift logits.
    candidates = torch.cat([candidates_logit, candidates_swift_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[tree_indices]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat(
        [
            tree_candidates,
            torch.zeros((1), dtype=torch.long, device=tree_candidates.device),
        ],
        dim=0,
    )

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = swift_logits[1]
        candidates_prob = torch.cat(
            [
                torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32),
                candidates_tree_prob.view(-1),
            ],
            dim=-1,
        )

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [
                tree_candidates_prob,
                torch.ones(
                    (1), dtype=torch.float32, device=tree_candidates_prob.device
                ),
            ],
            dim=0,
        )
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    swift_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.

    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - swift_position_ids (torch.Tensor): Positional IDs associated with the swift structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.

    Returns:
    - tuple: Returns swift logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the swift position IDs to the length of the input sequence.
    position_ids = swift_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates.
    outputs = model(
        tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    tree_logits = outputs.logits

    # Reorder the obtained logits and hidden states based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]
    
    model.model.swift_mask = None # reset attn mask
    return logits, outputs


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor,
    cart_candidates_prob,
    op,
    p_indices,
    tree_candidates,
    b_indices,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.
    - cart_candidates_prob (torch.Tensor): Cartesian candidates probabilities.
    - op (list): List of output probabilities.
    - p_indices (list): List of parent indices.
    - tree_candidates (torch.Tensor): Tree candidates.
    - b_indices (list): List of branch indices.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]
    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        adjustflag = False
        gtp = None
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = cart_candidates_prob[j, i]
                    if qx <= 0:
                        continue
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        q = op[i - 1][p_indices[j][i]].clone()
                        b = b_indices[j][i]
                        if len(b) > 0:
                            mask = tree_candidates[0][b]
                            q[mask] = 0
                            q = q / q.sum()
                        max_id = gtp.argmax()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        if torch.equal(
                            gtp.cpu(), torch.zeros(gtp.shape)
                        ):  # multinomial error
                            gtp[max_id] = 1
                        gtp = gtp / (gtp.sum() + 1e-6)
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        
        return best_candidate, accept_length - 1, sample_p


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token_num,
    past_key_values_data_list,
    current_length_data,
    sample_p,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token_num (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.
    - sample_p (torch.Tensor): Probability of the sampled token.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - new_token_num (int): Updated counter for the new tokens added.
    - sample_token (torch.Tensor): Token sampled from probability distribution.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [
            input_ids,
            candidates[None, best_candidate, : accept_length + 1].to(input_ids.device),
        ],
        dim=-1,
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[
            ..., select_indices.to(past_key_values_data.device), :
        ]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[
            ..., prev_input_len : prev_input_len + tgt.shape[-2], :
        ]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    prob = sample_p
    if logits_processor is not None:
        sample_token = torch.multinomial(prob, 1)
        sample_token = sample_token[None]
    else:
        sample_token = torch.argmax(prob)
        sample_token = sample_token[None, None]
    # Update the new token counter
    new_token_num += accept_length + 1

    return input_ids, new_token_num, sample_token


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
