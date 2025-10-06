import argparse
import copy
import logging
import random
import time

import numpy as np
import torch
from bayes_opt import BayesianOptimization, UtilityFunction
from fastchat.utils import str_to_torch_dtype
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

from evaluation.eval import reorg_answer_file, run_eval
from model.casspec.kv_cache import initialize_past_key_values
from model.casspec.modeling_llama import LlamaForCausalLM
from model.pld.pld import find_candidate_pred_tokens
from model.swift.utils import (
    evaluate_posterior,
    generate_swift_buffers,
    get_cache_configuration,
    get_choices_list,
    initialize_swift,
    prepare_logits_processor,
    reset_swift_mode,
    set_logger,
    swift_draft,
    swift_optimization,
    tree_decoding,
    update_inference_inputs,
)

first_acc_rates = []


def add_pld_paths_to_swift_buffers(
    swift_buffers,
    input_ids,
    sample_token,
    swift_logits,
    pld_max_ngram_size=3,
    pld_num_pred_tokens=7,
    max_pld_depths=3,
):
    """
    Add PLD candidate paths to existing SWIFT buffers with proper attention masks.
    """
    device = swift_buffers["tree_indices"].device
    retrieve_indices = swift_buffers["retrieve_indices"]
    tree_indices = swift_buffers["tree_indices"]
    swift_attn_mask = swift_buffers["swift_attn_mask"]
    swift_position_ids = swift_buffers["swift_position_ids"]

    logging.info(f"\n{'=' * 60}")
    logging.info("PLD AUGMENTATION START")
    logging.info(f"Input length: {input_ids.shape[1]}")
    logging.info(f"Original SWIFT paths: {retrieve_indices.shape[0]}")
    logging.info(f"Original tree size: {len(tree_indices)}")

    # Get SWIFT tokens
    swift_tokens = swift_logits[0]  # Shape: [num_draft_steps, TOPK]
    logging.info(f"SWIFT draft tokens shape: {swift_tokens.shape}")

    # Collect PLD sequences
    pld_sequences = []

    for depth in range(min(max_pld_depths, len(swift_tokens) + 1)):
        # Build context for PLD
        if depth == 0:
            context = input_ids
            prefix_indices = []
            logging.info(f"\nPLD at depth {depth} (from root):")
        else:
            # Build greedy path: sample_token + first choice at each subsequent level
            if depth == 1:
                greedy_tokens = sample_token[0]
            else:
                greedy_tokens = torch.cat(
                    [sample_token[0], swift_tokens[: depth - 1, 0]]
                )

            context = torch.cat([input_ids, greedy_tokens.unsqueeze(0)], dim=1)
            # Map to tree indices: [0 (root), 1 (first branch), ...]
            prefix_indices = [0] + list(range(1, depth + 1))
            logging.info(
                f"\nPLD at depth {depth} (after greedy path {prefix_indices}):"
            )
            logging.info(f"  Greedy tokens: {greedy_tokens.tolist()}")

        # Find PLD match
        pld_tokens = find_candidate_pred_tokens(
            context,
            max_ngram_size=pld_max_ngram_size,
            num_pred_tokens=pld_num_pred_tokens,
        )

        if len(pld_tokens) > 0:
            logging.info("  ✓ Found PLD match! Tokens: {pld_tokens.tolist()}")
            pld_sequences.append(
                {
                    "depth": depth,
                    "tokens": pld_tokens,
                    "prefix_indices": prefix_indices,
                    "prefix_length": depth,
                }
            )
        else:
            logging.info("  ✗ No PLD match found")

    if not pld_sequences:
        logging.info("No PLD sequences found. Returning original buffers.")
        logging.info(f"{'=' * 60}\n")
        return None, swift_buffers

    # --- Build enhanced structures ---
    original_tree_size = len(tree_indices)
    current_tree_size = original_tree_size

    # We'll append PLD tokens to tree_indices
    new_tree_indices = tree_indices.tolist()
    new_retrieve_indices = retrieve_indices.tolist()
    new_p_indices = swift_buffers["p_indices"].copy()
    new_b_indices = [
        bi.copy() if isinstance(bi, list) else bi for bi in swift_buffers["b_indices"]
    ]

    # Expand attention mask and position IDs
    new_attn_mask = swift_attn_mask.clone()
    new_position_ids = swift_position_ids.tolist()

    max_path_len = retrieve_indices.shape[1]

    for seq_idx, pld_seq in enumerate(pld_sequences):
        depth = pld_seq["depth"]
        tokens = pld_seq["tokens"]
        prefix_indices = pld_seq["prefix_indices"]

        logging.info(f"\nAdding PLD sequence {seq_idx} at depth {depth}:")
        logging.info(f"  Tokens to add: {tokens.tolist()}")
        logging.info(f"  Prefix indices: {prefix_indices}")

        # Add each PLD token to tree
        pld_token_indices = []
        for i, token in enumerate(tokens):
            token_idx = current_tree_size
            new_tree_indices.append(token.item())
            pld_token_indices.append(token_idx)

            # Position ID = depth + i + 1
            position = depth + i + 1
            new_position_ids.append(position)

            current_tree_size += 1

        logging.info(f"  Assigned tree indices: {pld_token_indices}")

        # Create retrieve path: prefix + pld_tokens
        retrieve_path = prefix_indices + pld_token_indices

        # Pad to max length
        if len(retrieve_path) < max_path_len:
            retrieve_path = retrieve_path + [-1] * (max_path_len - len(retrieve_path))
        else:
            retrieve_path = retrieve_path[:max_path_len]

        new_retrieve_indices.append(retrieve_path)
        logging.info(f"  Retrieve path: {retrieve_path}")

        # Add p_indices and b_indices (simplified - no parent blocking for now)
        new_p_indices.append([-1] + [0] * (max_path_len - 1))
        new_b_indices.append([[] for _ in range(max_path_len)])

    # --- Rebuild attention mask ---
    total_size = current_tree_size
    new_attn_mask_expanded = torch.zeros(
        (1, 1, total_size, total_size), device=device, dtype=swift_attn_mask.dtype
    )

    # Copy original mask
    new_attn_mask_expanded[:, :, :original_tree_size, :original_tree_size] = (
        new_attn_mask
    )

    # Add attention for PLD tokens
    for seq_idx, pld_seq in enumerate(pld_sequences):
        prefix_indices = pld_seq["prefix_indices"]
        depth = pld_seq["depth"]
        num_tokens = len(pld_seq["tokens"])

        # PLD tokens start at original_tree_size + offset
        offset = sum(len(s["tokens"]) for s in pld_sequences[:seq_idx])
        pld_start = original_tree_size + offset

        for i in range(num_tokens):
            token_pos = pld_start + i

            # Can attend to root (position 0)
            new_attn_mask_expanded[0, 0, token_pos, 0] = 1

            # Can attend to prefix (ancestors in the path)
            for ancestor_idx in prefix_indices:
                new_attn_mask_expanded[0, 0, token_pos, ancestor_idx] = 1

            # Can attend to previous PLD tokens in same sequence
            for j in range(i):
                new_attn_mask_expanded[0, 0, token_pos, pld_start + j] = 1

            # Can attend to itself
            new_attn_mask_expanded[0, 0, token_pos, token_pos] = 1

    # Convert back to tensors
    enhanced_buffers = {
        "swift_attn_mask": new_attn_mask_expanded,
        "tree_indices": torch.tensor(new_tree_indices, dtype=torch.long, device=device),
        "swift_position_ids": torch.tensor(
            new_position_ids, dtype=torch.long, device=device
        ),
        "retrieve_indices": torch.tensor(
            new_retrieve_indices, dtype=torch.long, device=device
        ),
        "p_indices": new_p_indices,
        "b_indices": new_b_indices,
    }

    logging.info(f"\n{'=' * 60}")
    logging.info("ENHANCED BUFFERS SUMMARY:")
    logging.info(f"  Tree size: {original_tree_size} → {current_tree_size}")
    logging.info(
        f"  Retrieve paths: {retrieve_indices.shape[0]} → {len(new_retrieve_indices)}"
    )
    logging.info(
        f"  Attention mask shape: {swift_attn_mask.shape} → {new_attn_mask_expanded.shape}"
    )
    logging.info(f"  Added {len(pld_sequences)} PLD sequences")
    logging.info(f"{'=' * 60}\n")

    # Also need to return enhanced candidates
    # Build the full candidate tensor
    base_candidates = torch.cat([sample_token[0], swift_tokens.view(-1)], dim=0)

    pld_all_tokens = torch.cat([s["tokens"] for s in pld_sequences], dim=0)
    enhanced_candidates = torch.cat([base_candidates, pld_all_tokens], dim=0)

    logging.info(
        f"Enhanced candidates size: {len(base_candidates)} → {len(enhanced_candidates)}"
    )

    return enhanced_candidates, enhanced_buffers


def generate_candidates_with_pld(
    swift_logits,
    tree_indices,
    retrieve_indices,
    sample_token,
    logits_processor,
    enhanced_candidates=None,
):
    """
    Modified version of generate_candidates that uses enhanced candidates if available.
    """
    sample_token = sample_token.to(tree_indices.device)

    logging.info("\nGENERATE_CANDIDATES_WITH_PLD:")
    logging.info(f"  Tree indices size: {len(tree_indices)}")
    logging.info(f"  Retrieve indices shape: {retrieve_indices.shape}")

    if enhanced_candidates is not None:
        candidates = enhanced_candidates
        logging.info(f"  Using ENHANCED candidates: {len(candidates)} tokens")
        logging.info(f"  Candidates: {candidates.tolist()}")
    else:
        candidates_logit = sample_token[0]
        candidates_swift_logits = swift_logits[0]
        candidates = torch.cat(
            [candidates_logit, candidates_swift_logits.view(-1)], dim=-1
        )
        logging.info(f"  Using ORIGINAL candidates: {len(candidates)} tokens")

    # Validate indices
    max_candidate_idx = len(candidates) - 1
    if tree_indices.max().item() > max_candidate_idx:
        logging.error(
            f"  ERROR: tree_indices max ({tree_indices.max().item()}) > candidates max index ({max_candidate_idx})"
        )
        logging.error(f"  Tree indices: {tree_indices.tolist()}")
        # Clamp invalid indices
        tree_indices = torch.clamp(tree_indices, 0, max_candidate_idx)

    # Map to tree structure
    tree_candidates = candidates[tree_indices]
    logging.info(f"  Tree candidates: {tree_candidates.tolist()}")

    tree_candidates_ext = torch.cat(
        [
            tree_candidates,
            torch.zeros((1), dtype=torch.long, device=tree_candidates.device),
        ],
        dim=0,
    )

    # Validate retrieve_indices
    max_tree_idx = len(tree_candidates_ext) - 1
    retrieve_indices_clamped = retrieve_indices.clone()
    retrieve_indices_clamped[retrieve_indices_clamped > max_tree_idx] = max_tree_idx
    retrieve_indices_clamped[retrieve_indices_clamped < -1] = -1

    # Retrieve cartesian candidates
    cart_candidates = tree_candidates_ext[retrieve_indices_clamped]
    logging.info(f"  Cart candidates shape: {cart_candidates.shape}")
    logging.info(f"  Sample cart candidates:\n{cart_candidates[:3]}")

    # Handle probabilities
    if logits_processor is not None:
        candidates_tree_prob = swift_logits[1]
        candidates_prob = torch.cat(
            [
                torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32),
                candidates_tree_prob.view(-1),
            ],
            dim=-1,
        )

        # Extend with probs for PLD tokens
        if enhanced_candidates is not None:
            pld_token_count = len(enhanced_candidates) - len(candidates_prob)
            if pld_token_count > 0:
                # Assign moderate confidence to PLD tokens
                pld_probs = (
                    torch.ones(
                        pld_token_count,
                        device=candidates_prob.device,
                        dtype=torch.float32,
                    )
                    * 0.7
                )
                candidates_prob = torch.cat([candidates_prob, pld_probs], dim=0)
                logging.info(f"  Extended probs for {pld_token_count} PLD tokens")

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
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices_clamped]
    else:
        cart_candidates_prob = None

    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def swift_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    statistics=None,
    optimizer=None,
    utility=None,
    logits_processor=None,
    max_steps=512,
    use_pld=True,
    pld_max_ngram_size=3,
    pld_num_pred_tokens=7,
    max_pld_depths=2,
):
    """
    Enhanced SWIFT forward with PLD integration.
    """
    input_ids = inputs.input_ids.cuda()
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_swift_mode(model)

    start_prefill = time.time()
    swift_logits, sample_token, top1_prob = initialize_swift(
        input_ids,
        model,
        max_new_tokens,
        past_key_values,
        past_key_values_data,
        current_length_data,
        logits_processor=logits_processor,
    )
    logging.info(f"Prefill time: {time.time() - start_prefill:.4f}s")

    # Clone for optimization
    input_past_key_values_data = [kv.clone() for kv in past_key_values_data]
    input_current_length_data = current_length_data.clone()

    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    pld_contribution = 0
    pld_attempts = 0

    timings = {
        "total_step": [],
        "draft_loop": [],
        "avg_draft_time": [],
        "verify": [],
        "accept_update": [],
        "misc_overhead": [],
        "pld_augmentation": [],
    }
    step_end_time = time.time()

    for idx in range(max_steps):
        logging.info(f"\n{'#' * 80}")
        logging.info(f"STEP {idx}")
        logging.info(f"{'#' * 80}")

        start_step = time.time()
        timings["misc_overhead"].append(start_step - step_end_time)

        draft_token_num += len(top1_prob)

        # Initialize SWIFT buffer
        swift_choices = eval(
            f"{get_choices_list(top1_prob, logits_processor=logits_processor)}"
        )
        swift_buffers = generate_swift_buffers(
            swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device
        )

        logging.info("\nOriginal SWIFT structure:")
        logging.info(f"  Choices: {swift_choices[:5]}...")  # Show first 5
        logging.info(f"  Tree size: {len(swift_buffers['tree_indices'])}")
        logging.info(f"  Paths: {swift_buffers['retrieve_indices'].shape[0]}")

        # --- PLD AUGMENTATION ---
        pld_start = time.time()
        enhanced_candidates = None
        active_buffers = swift_buffers

        if use_pld and input_ids.shape[1] >= pld_max_ngram_size:
            enhanced_candidates, enhanced_buffers = add_pld_paths_to_swift_buffers(
                swift_buffers,
                input_ids,
                sample_token,
                swift_logits,
                pld_max_ngram_size=pld_max_ngram_size,
                pld_num_pred_tokens=pld_num_pred_tokens,
                max_pld_depths=max_pld_depths,
            )

            if enhanced_candidates is not None:
                pld_attempts += 1
                active_buffers = enhanced_buffers
                # Update model buffers
                model.swift_buffers = enhanced_buffers
                model.model.swift_mask = enhanced_buffers["swift_attn_mask"]
                logging.info("✓ Using PLD-enhanced buffers")
            else:
                model.swift_buffers = swift_buffers
                model.model.swift_mask = swift_buffers["swift_attn_mask"]
                logging.info("✗ No PLD enhancement, using original SWIFT")
        else:
            model.swift_buffers = swift_buffers
            model.model.swift_mask = swift_buffers["swift_attn_mask"]

        model.swift_choices = swift_choices

        timings["pld_augmentation"].append(time.time() - pld_start)

        # Generate candidates
        candidates, cart_candidates_prob, tree_candidates = (
            generate_candidates_with_pld(
                swift_logits,
                active_buffers["tree_indices"],
                active_buffers["retrieve_indices"],
                sample_token,
                logits_processor,
                enhanced_candidates=enhanced_candidates,
            )
        )

        logging.info("\nTree decoding with:")
        logging.info(f"  Tree candidates shape: {tree_candidates.shape}")
        logging.info(f"  Position IDs: {active_buffers['swift_position_ids'].tolist()}")
        logging.info(
            f"  Attention mask: {active_buffers['swift_attn_mask']}"
        )

        # Tree decoding (verification)
        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            active_buffers["swift_position_ids"],  # Use active position IDs!
            input_ids,
            active_buffers["retrieve_indices"],
        )

        verify_end_time = time.time()
        timings["verify"].append(verify_end_time - pld_start)

        logging.info("\nEvaluating posterior:")
        logging.info(f"  Logits shape: {logits.shape}")
        logging.info(f"  Candidates shape: {candidates.shape}")
        logging.info(f"  Number of paths: {candidates.shape[0]}")

        # Evaluate posterior
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits,
            candidates,
            logits_processor,
            cart_candidates_prob,
            swift_logits[2],
            active_buffers["p_indices"],
            tree_candidates,
            active_buffers["b_indices"],
        )

        logging.info("\nPosterior result:")
        logging.info(f"  Best candidate: {best_candidate}")
        logging.info(f"  Accept length: {accept_length}")
        logging.info(
            f"  Accepted path: {candidates[best_candidate, : accept_length + 1].tolist()}"
        )

        # Check if PLD contributed
        original_path_count = swift_buffers["retrieve_indices"].shape[0]
        if enhanced_candidates is not None and best_candidate >= original_path_count:
            pld_contribution += accept_length + 1
            logging.info(f"{'*' * 60}")
            logging.info("✓✓✓ PLD PATH ACCEPTED! ✓✓✓")
            logging.info(
                f"  Path index: {best_candidate} (PLD path {best_candidate - original_path_count})"
            )
            logging.info(f"  Tokens accepted: {accept_length + 1}")
            logging.info(
                f"  Path: {candidates[best_candidate, : accept_length + 1].tolist()}"
            )
            logging.info(f"{'*' * 60}")

        if accept_length == 0:
            first_acc_rates.append(0)
        else:
            first_acc_rates.append(1)

        # Update inputs
        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            active_buffers["retrieve_indices"],
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p,
        )
        accept_update_end_time = time.time()
        timings["accept_update"].append(accept_update_end_time - verify_end_time)

        # Layer set optimization
        if (
            new_token_num > (statistics["context_window"] + 1)
            and statistics["optimization"]
            and idx % statistics["opt_interval"] == 0
        ):
            logging.info("Swift optimization" + "-" * 10)
            swift_optimization(
                model,
                input_ids[:, input_len:],
                input_past_key_values_data,
                input_current_length_data,
                new_token_num,
                statistics,
                optimizer=optimizer,
                utility=utility,
            )

        # SWIFT drafting
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )

        draft_loop_end_time = time.time()
        draft_time = draft_loop_end_time - accept_update_end_time
        timings["draft_loop"].append(draft_time)
        timings["avg_draft_time"].append(
            draft_time / len(top1_prob) if len(top1_prob) > 0 else 0
        )

        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token_num > max_new_tokens:
            break

        step_end_time = time.time()
        timings["total_step"].append(step_end_time - start_step)

    total_acc_rate = total_acc_num / draft_token_num if draft_token_num > 0 else 0
    logging.info(f"\n{'=' * 80}")
    logging.info("FINAL STATISTICS")
    logging.info(f"{'=' * 80}")
    logging.info(f"Total acceptance rate: {total_acc_rate:.4f}")
    logging.info(f"PLD attempts: {pld_attempts}")
    logging.info(
        f"PLD contribution: {pld_contribution}/{new_token_num} tokens ({pld_contribution / new_token_num * 100:.2f}%)"
    )
    logging.info(f"Total steps: {idx}")

    logging.info("\n--- Performance Timings (Average per Step) ---")
    for key, values in timings.items():
        if values:
            avg_time = np.mean(values)
            logging.info(f"{key}: {avg_time:.4f}s")

    logging.info(f"Average Acceptance Length: {np.mean(accept_length_list) - 1:.4f}")
    logging.info(f"Finished. Total steps: {idx}, Total generated: {new_token_num}")

    return input_ids, new_token_num, idx + 1, accept_length_list


def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1034,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for swift sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
    )
    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.45,
        help="The skipped layer ratio of swift.",
    )
    parser.add_argument(
        "--opt-interval",
        type=int,
        default=1,
        help="The interval of swift optimization.",
    )
    parser.add_argument(
        "--bayes-interval",
        type=int,
        default=25,
        help="The interval of bayesian optimization.",
    )
    parser.add_argument(
        "--max-opt-iter",
        type=int,
        default=1000,
        help="The maximum layer set optimization iteration.",
    )
    parser.add_argument(
        "--max-tolerance-iter",
        type=int,
        default=300,
        help="The maximum tolerance of layer set search iteration.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="The early stop threshold of layer set search.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="The context window of swift.",
    )
    parser.add_argument(
        "--optimization",
        action="store_true",
        default=False,
        help="Layer set optimization.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        default=False,
        help="Bayes Optimization of Layer set.",
    )
    parser.add_argument(
        "--cache-hit",
        action="store_true",
        default=False,
        help="Whether to use cached SWIFT configuration.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The sampling seed.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--use-pld",
        action="store_true",
        default=False,
        help="Enable PLD candidate augmentation",
    )
    parser.add_argument(
        "--pld-max-ngram", type=int, default=3, help="Max n-gram size for PLD"
    )
    parser.add_argument(
        "--pld-num-tokens", type=int, default=7, help="Number of tokens PLD predicts"
    )
    parser.add_argument(
        "--pld-max-depths",
        type=int,
        default=2,
        help="Maximum tree depths to add PLD paths",
    )

    args = parser.parse_args()

    args.model_name = (
        args.model_id
        + "-casspec-"
        + str(args.dtype)
        + "-temp-"
        + str(args.temperature)
        + "-top-p-"
        + str(args.top_p)
        + "-seed-"
        + str(args.seed)
        + "-max_new_tokens-"
        + str(args.max_new_tokens)
        + "-opt_interval-"
        + str(args.opt_interval)
        + "-bayes_interval-"
        + str(args.bayes_interval)
        + "-max_opt-"
        + str(args.max_opt_iter)
        + "-max_tolerance-"
        + str(args.max_tolerance_iter)
        + "-max_score-"
        + str(args.max_score)
        + "-context_window-"
        + str(args.context_window)
        + "-skip_ratio-"
        + str(args.skip_ratio)
        # + "-pld-" if args.use_pld else ""
        # + f"maxngram{args.pld_max_ngram}-numtok{args.pld_num_tokens}-maxdepth{args.pld_max_depths}" if args.use_pld else ""
    )
    answer_file = f"data/{args.bench_name}/model_answer/{args.model_name}.jsonl"
    set_logger()

    print(f"Output to {answer_file}")

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file

    torch.nn.Linear.reset_parameters = lambda x: None

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(
            temperature=args.temperature, top_p=args.top_p
        )
    else:
        logits_processor = None

    if args.cache_hit:
        # Load the cached layer set configuration
        args.optimization, args.bayes = False, False
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = get_cache_configuration(
            model_name=args.model_id, bench_name=args.bench_name
        )
    else:
        # Unified layer set initialization
        _attn_skip_layer_id_set = np.arange(
            1, model.config.num_hidden_layers - 1, 2
        )  # keep the first and last layer
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)

    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    seed_everything(args.seed)
    # Bayes Optimization Settings
    pbounds = {
        f"x{i}": (0, 1) for i in range((model.config.num_hidden_layers - 2) * 2)
    }  # keep the first and last layer
    optimizer = BayesianOptimization(
        f=None, pbounds=pbounds, random_state=1, verbose=1, allow_duplicate_points=True
    )
    optimizer.set_gp_params(alpha=1e-2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    statistics = {
        "origin_score": 0,
        "opt_iter": 0,
        "tolerance_iter": 0,
        "skip_ratio": args.skip_ratio,
        "acceptance_rate_list": [],
        "opt_interval": args.opt_interval,
        "bayes_interval": args.bayes_interval,
        "max_opt_iter": args.max_opt_iter,
        "max_tolerance_iter": args.max_tolerance_iter,
        "max_score": args.max_score,
        "context_window": args.context_window,
        "optimization": args.optimization,
        "bayes": args.bayes,
    }

    def swift_forward_wrapper(inputs, model, tokenizer, max_new_tokens, **kwargs):
        return swift_forward(
            inputs,
            model,
            tokenizer,
            max_new_tokens,
            use_pld=args.use_pld,
            pld_max_ngram_size=args.pld_max_ngram,
            pld_num_pred_tokens=args.pld_num_tokens,
            max_pld_depths=args.pld_max_depths,
            **kwargs,
        )

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=swift_forward_wrapper,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        # data_num=args.data_num,
        # seed=args.seed,
        optimizer=optimizer,
        utility=utility,
        statistics=statistics,
        logits_processor=logits_processor,
    )

    print("First acceptance rate: ", np.mean(first_acc_rates))

    reorg_answer_file(answer_file)
